"""UniRef50 + Swiss-Prot sequences and concepts inspired by CB-Lobster.

Unlike the original, descriptor categories are extracted from the merged local
records, without Lobster's fixed vocabularies or frequency filtering. This
covers the labels present in the imported data as UniProt releases change.
Category names, indices and one-hot width depend on the imported records
(including max_sequences) and are saved in the cache. The 14 Biopython
properties are stored in their original units, without fixed normalization.

Reference: https://arxiv.org/abs/2411.06090 (Appendix B.1).
"""

import gzip
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from itertools import islice
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch
from tqdm import tqdm

from ...annotations import Annotations
from ...tensor import AnnotatedTensor
from ..base.dataset import ConceptDataset
from ..io import download_url


SOURCE_URLS = {
    "uniprot_sprot.dat.gz": (
        "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
        "knowledgebase/complete/uniprot_sprot.dat.gz"
    ),
    "uniref50.xml.gz": (
        "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.xml.gz"
    ),
}
DESCRIPTOR_FIELDS = (
    "cluster_name", "biological_process", "cellular_component",
    "molecular_function", "organism", "taxon",
)
GO_FIELDS = {
    "P": "biological_process",
    "C": "cellular_component",
    "F": "molecular_function",
}
BIOPYTHON_CONCEPTS = [
    "molecular_weight", "aromaticity", "instability_index", "isoelectric_point",
    "gravy", "charge_at_pH6", "charge_at_pH7", "helix_fraction",
    "turn_structure_fraction", "sheet_structure_fraction",
    "molar_extinction_coefficient_reduced", "molar_extinction_coefficient_oxidized",
    "avg_hydrophilicity", "avg_surface_accessibility",
]


def _swissprot_records(handle):
    """Read one Swiss-Prot protein at a time as (sequence, annotation fields)."""
    for line in handle:
        if not line.strip():
            continue
        if not line.startswith("ID   "):
            raise ValueError(f"Invalid Swiss-Prot record: {line.strip()}")
        entry_id = line.split()[1]
        sequence, organism = [], []
        accession = None
        terms = {name: set() for name in GO_FIELDS.values()}
        for line in handle:
            tag, value = line[:2], line[5:].strip()
            if tag == "//":
                break
            if tag == "AC" and not accession:
                accession = value.split(";", 1)[0].strip()
            elif tag == "OS":
                organism.append(value)
            elif tag == "DR" and value.startswith("GO;"):
                _, go_id, term, *_ = (part.strip() for part in value.split(";"))
                aspect, term = term.split(":", 1)
                if aspect in GO_FIELDS:
                    terms[GO_FIELDS[aspect]].add(f"{term} [{go_id}]")
            elif tag == "  ":
                sequence.append("".join(value.split()))
        else:
            raise ValueError("Incomplete Swiss-Prot record: missing //.")
        fields = {name: "; ".join(sorted(values)) for name, values in terms.items()}
        fields["organism"] = " ".join(organism).removesuffix(".")
        if not accession:
            raise ValueError(f"Swiss-Prot record {entry_id} has no accession.")
        fields["swissprot_accession"] = accession
        yield "".join(sequence), fields


def _uniref_records(handle):
    """Stream XML members as (accessions, cluster fields), including representatives.

    Release each member immediately: even a very large cluster need not fit
    in memory. UniParc-only members have no UniProtKB accession to match.
    """
    context = ElementTree.iterparse(handle, events=("start", "end"))
    _, root = next(context)
    if root.tag.rsplit("}", 1)[-1] not in {"UniRef50", "UniRef"}:
        raise ValueError("Expected a UniRef50 XML document.")
    cluster = None
    for event, element in context:
        tag = element.tag.rsplit("}", 1)[-1]
        if event == "start" and tag == "entry":
            cluster = element
            cluster_id = cluster.attrib["id"]
            if not cluster_id.startswith("UniRef50_"):
                raise ValueError(f"Expected a UniRef50 cluster: {cluster_id}")

        elif event == "end" and tag in {"representativeMember", "member"}:
            # The member is complete; look up its accessions and cluster fields.
            taxon = cluster.find("{*}property[@type='common taxon']")
            fields = {
                "uniref_cluster_id": cluster_id,
                "cluster_name": cluster.findtext("{*}name") or None,
                "taxon": taxon.get("value") if taxon is not None else None,
            }
            reference = element.find("{*}dbReference")
            if reference is None:
                raise ValueError("UniRef member has no database reference.")
            accessions = tuple(
                prop.attrib["value"]
                for prop in reference.findall("{*}property[@type='UniProtKB accession']")
            )

            # Release the member before yielding, keeping only the extracted data.
            cluster.remove(element)
            element.clear()
            if accessions:
                yield accessions, fields

        elif event == "end" and tag == "entry":
            root.remove(cluster)
            cluster.clear()


def _biopython_concepts(sequence):
    """Calculate the 14 protein properties in their original units."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis, ProtParamData

    # Lobster substitutes nonstandard residues when calculating descriptors.
    sequence = "".join(aa if aa in ProtParamData.hw else "A" for aa in sequence)
    protein = ProteinAnalysis(sequence)
    values = [
        protein.molecular_weight(), protein.aromaticity(),
        protein.instability_index(), protein.isoelectric_point(), protein.gravy(),
        protein.charge_at_pH(6), protein.charge_at_pH(7),
        *protein.secondary_structure_fraction(),
        *protein.molar_extinction_coefficient(),
        sum(ProtParamData.hw[aa] for aa in sequence) / len(sequence),
        sum(ProtParamData.em[aa] for aa in sequence) / len(sequence),
    ]
    return torch.tensor(values, dtype=torch.float32)


class _BatchTensor(AnnotatedTensor):
    """Avoid pickle recursion when workers send annotated batches."""

    def __reduce__(self):
        return AnnotatedTensor, (self.tensor, self.annotations, self.axis)


class UniRefSwissProtDataset(ConceptDataset):
    """UniRef50 + Swiss-Prot with sequences and concept labels stored on disk.

    Swiss-Prot is imported first, keeping each accession and its organism/GO
    annotations even when different records have identical sequences.
    The UniRef50 XML then supplies cluster metadata by matching the
    Swiss-Prot primary accession against representative and member
    accessions. Only Swiss-Prot records matched to UniRef50 are retained
    when include_uniref is True. UniRef50 does not add protein samples.
    This enrichment is a local policy; the paper's original join is not published.
    SQLite stores sequences, descriptor fields and Biopython properties.
    Input record IDs and concepts are ordinary memory-mapped
    tensors. __getitem__ reads the sequence for an ID and keeps the standard
    sample structure. One-hot width depends on the categories found in the
    merged records; categorical storage requires 20 floats per protein.
    The full table need not fit in RAM.

    Swiss-Prot supplies GO annotations and organism; UniRef50 supplies cluster
    name and common taxon. Gzip DAT records and XML members are streamed;
    Biopython is used only to calculate protein properties. The accession-to-row
    lookup uses RAM proportional to the selected Swiss-Prot proteins.
    Descriptor vocabularies contain all distinct nonempty field values in the
    merged records, with UNKNOWN reserved for missing values. Complete GO
    combinations remain single categories. Continuous properties retain their
    original units; optional scaling belongs to the datamodule's training split.
    Downloads use current UniProt releases.

    The first build requires disk space and time proportional to the sources;
    later runs open the index directly. Each DataLoader worker opens its own
    read-only connection. Bulk embedding precomputation follows the base
    class's in-memory behavior; for full training keep the backbone in the
    model and apply it per batch. UniRefSwissProtDataModule uses the library's
    standard splitter and DataLoaders, with split and shuffle indices in RAM.

    Args:
        root: Raw files and SQLite cache, default ./data/uniref_swissprot.
        max_length: Length used for inputs and calculated properties.
        max_sequences: Optional Swiss-Prot record limit; None reads all.
            Does not limit raw download sizes.
        concept_subset: Optional concept names to retain.
        include_uniref: True retains only Swiss-Prot records matched to
            UniRef50. False reads all selected Swiss-Prot records.
        one_hot_encoding: True returns 14 continuous properties followed by
            one binary indicator per descriptor category. False returns the
            same properties followed by six class indices. Groups with only
            UNKNOWN are constant zero columns, annotated as binary.
        max_uniref_members: Optional limit on UniRef50 representatives and
            members with UniProtKB accessions; None scans all. Only Swiss-Prot
            matches found in this prefix are retained. Raw downloads remain full.
    """

    def __init__(
        self,
        root: str | None = None,
        max_length: int = 512,
        max_sequences: int | None = None,
        concept_subset: list | None = None,
        include_uniref: bool = True,
        one_hot_encoding: bool = True,
        max_uniref_members: int | None = None,
    ):
        if max_length < 1 or (max_sequences is not None and max_sequences < 1):
            raise ValueError("max_length and max_sequences must be positive.")
        if max_uniref_members is not None and max_uniref_members < 1:
            raise ValueError("max_uniref_members must be positive or None.")
        self.root = os.fspath(root) if root is not None else "./data/uniref_swissprot"
        self.max_length = max_length
        self.max_sequences = max_sequences
        self.max_uniref_members = max_uniref_members
        self.include_uniref = include_uniref
        self.one_hot_encoding = one_hot_encoding
        self._connection = None
        self._connection_pid = None

        inputs, concepts, annotations, graph = self.load()
        # The base casts inputs to float32. Validate the count without allocating
        # features, then attach the integer IDs to preserve their dtype and mapping.
        super().__init__(
            input_data=torch.empty((len(inputs), 0)),
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,
            reorder_by_type=False,
        )
        self.input_data = inputs
        print(f"Final dataset: {len(self):,} Swiss-Prot records.")

    @property
    def raw_filenames(self):
        return [
            name for name in SOURCE_URLS
            if self.include_uniref or name != "uniref50.xml.gz"
        ]

    @property
    def processed_filenames(self):
        source = "uniref_swissprot_matched" if self.include_uniref else "swissprot"
        # Separate per-accession records from older sequence-deduplicated caches.
        name = f"{source}_records_len{self.max_length}_n{self.max_sequences}"
        if self.include_uniref and self.max_uniref_members is not None:
            name += f"_members{self.max_uniref_members}"
        mode = "onehot" if self.one_hot_encoding else "categorical"
        return [f"{name}.sqlite", f"{name}.ids.i64", f"{name}.{mode}.f32",
                f"{name}.{mode}.annotations.pt"]

    def download(self):
        for filename in self.raw_filenames:
            target = os.path.join(self.root_dir, filename)
            if os.path.exists(target):
                continue
            # An interrupted download must never appear as a complete raw source.
            with tempfile.TemporaryDirectory(dir=self.root_dir) as staging:
                source = download_url(SOURCE_URLS[filename], staging, filename=filename)
                os.replace(source, target)

    def build(self):
        """Import Swiss-Prot, retain UniRef matches, then save concepts."""
        database = self.processed_paths[0]
        if os.path.exists(database):
            # Reuse prepared proteins when only the concept encoding is missing.
            self._save_concepts()
            return

        self.maybe_download()
        with tempfile.TemporaryDirectory(dir=self.root_dir) as staging:
            temporary = os.path.join(staging, "proteins.sqlite")
            with closing(sqlite3.connect(temporary)) as connection:
                self._initialize_sequence_table(connection)
                swissprot_count = self._import_swissprot(connection)
                print(f"Swiss-Prot read: {swissprot_count:,} records.")
                if self.include_uniref:
                    matched_count = self._merge_uniref(connection)
                    initial_ratio = matched_count / swissprot_count if swissprot_count else 0
                    print(f"Swiss-Prot retained after UniRef merge: "
                          f"{matched_count:,}/{swissprot_count:,} records ({initial_ratio:.2%}).")
                self._extract_biopython_concepts(connection)
                connection.commit()
            os.replace(temporary, database)
        self._save_concepts()

    def _initialize_sequence_table(self, connection):
        """Create one row per Swiss-Prot accession, allowing identical sequences."""
        connection.execute("""
            CREATE TABLE proteins (
                id INTEGER PRIMARY KEY, sequence TEXT NOT NULL, properties BLOB,
                swissprot_accession TEXT UNIQUE NOT NULL, cluster_name TEXT,
                biological_process TEXT, cellular_component TEXT,
                molecular_function TEXT, organism TEXT, taxon TEXT
            )
        """)

    def _import_swissprot(self, connection):
        """Load Swiss-Prot records, preserving accessions with identical sequences."""
        path = os.path.join(self.root_dir, "uniprot_sprot.dat.gz")
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            proteins = islice(_swissprot_records(handle), self.max_sequences)
            for count, (sequence, fields) in enumerate(
                tqdm(proteins, desc="Importing Swiss-Prot", unit="proteins"), start=1
            ):
                sequence = "".join(sequence.split()).upper()
                if not sequence:
                    continue
                connection.execute("""
                    INSERT INTO proteins (
                        sequence, swissprot_accession,
                        biological_process, cellular_component, molecular_function, organism
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (sequence[:self.max_length], fields["swissprot_accession"],
                      fields["biological_process"] or None,
                      fields["cellular_component"] or None,
                      fields["molecular_function"] or None, fields["organism"] or None))
        return count

    def _merge_uniref(self, connection):
        """Enrich Swiss-Prot by exact accession membership, removing unmatched rows."""
        row_ids = dict(connection.execute("SELECT swissprot_accession, id FROM proteins"))
        matched_clusters = {}
        path = os.path.join(self.root_dir, "uniref50.xml.gz")
        with gzip.open(path, "rb") as handle:
            members = islice(_uniref_records(handle), self.max_uniref_members)
            for accessions, fields in tqdm(members, total=self.max_uniref_members,
                                           desc="Reading UniRef50 members", unit="members"):
                for accession in accessions:
                    row_id = row_ids.get(accession)
                    if row_id is None:
                        continue
                    cluster_id = fields["uniref_cluster_id"]
                    previous = matched_clusters.get(row_id)
                    if previous is not None and previous != cluster_id:
                        raise ValueError(f"Swiss-Prot accession {accession} occurs in multiple "
                                         f"UniRef50 clusters: {previous}, {cluster_id}")
                    matched_clusters[row_id] = cluster_id
                    connection.execute("""
                        UPDATE proteins SET cluster_name = ?, taxon = ? WHERE id = ?
                    """, (fields["cluster_name"], fields["taxon"], row_id))
        connection.executemany(
            "DELETE FROM proteins WHERE id = ?",
            ((row_id,) for row_id in row_ids.values() if row_id not in matched_clusters),
        )
        return len(matched_clusters)

    def _extract_biopython_concepts(self, connection):
        """Calculate properties and collect categories from the merged records."""
        count = connection.execute("SELECT COUNT(*) FROM proteins").fetchone()[0]
        if not count:
            raise ValueError("No protein sequences found in the source files.")
        vocabularies = []
        for name in DESCRIPTOR_FIELDS:
            values = connection.execute(
                f"SELECT DISTINCT {name} FROM proteins "
                f"WHERE {name} IS NOT NULL AND {name} NOT IN ('', 'UNKNOWN') ORDER BY {name}"
            )
            vocabularies.append(["UNKNOWN"] + [value for (value,) in values])

        rows = connection.execute("SELECT id, sequence FROM proteins ORDER BY id")
        for row_id, sequence in tqdm(rows, total=count, desc="Calculating Biopython concepts"):
            properties = _biopython_concepts(sequence).numpy().astype("<f4", copy=False).tobytes()
            connection.execute("UPDATE proteins SET properties = ? WHERE id = ?", (properties, row_id))
        connection.execute("CREATE TABLE metadata (value TEXT NOT NULL)")
        connection.execute("INSERT INTO metadata VALUES (?)", (json.dumps({
            "count": count, "vocabularies": vocabularies,
        }),))

    def _concept_annotations(self, vocabularies):
        """Describe the columns for the selected concept encoding."""
        descriptors = [f"{name} {value}" for name, values in zip(DESCRIPTOR_FIELDS, vocabularies)
                       for value in values]
        labels = BIOPYTHON_CONCEPTS + (descriptors if self.one_hot_encoding else list(DESCRIPTOR_FIELDS))
        return Annotations(
            labels=labels,
            cardinalities=[1] * 14 + ([1] * len(descriptors) if self.one_hot_encoding
                                      else [len(values) for values in vocabularies]),
            types=["continuous"] * 14 + (["binary"] * len(descriptors)
                                         if self.one_hot_encoding else [
                                             "categorical" if len(values) > 1 else "binary"
                                             for values in vocabularies
                                         ]),
            states=None if self.one_hot_encoding else [["0"] for _ in range(14)] + vocabularies,
        )

    def _save_concepts(self):
        """Write mapped concept rows and their annotations in the selected encoding."""
        database, id_path, concept_path, annotation_path = self.processed_paths
        with closing(sqlite3.connect(database)) as connection:
            metadata = json.loads(connection.execute("SELECT value FROM metadata").fetchone()[0])
            vocabularies = metadata["vocabularies"]
            annotations = self._concept_annotations(vocabularies)
            lookups = [{value: i for i, value in enumerate(values)} for values in vocabularies]
            offsets = np.cumsum([0, *(len(values) for values in vocabularies[:-1])])
            with tempfile.TemporaryDirectory(dir=self.root_dir) as staging:
                ids = os.path.join(staging, "ids.i64")
                concepts = os.path.join(staging, "concepts.f32")
                rows = connection.execute(
                    f"SELECT id, properties, {', '.join(DESCRIPTOR_FIELDS)} FROM proteins ORDER BY id"
                )
                with open(ids, "wb") as id_file, open(concepts, "wb") as concept_file:
                    for row_id, properties, *fields in tqdm(rows, total=metadata["count"], desc="Writing concepts"):
                        states = [lookup.get(value, lookup["UNKNOWN"])
                                  for lookup, value in zip(lookups, fields)]
                        values = np.zeros(len(annotations.labels), dtype="<f4")
                        values[:14] = np.frombuffer(properties, dtype="<f4")
                        if self.one_hot_encoding:
                            values[14 + offsets + states] = 1
                        else:
                            values[14:] = states
                        id_file.write(row_id.to_bytes(8, "little", signed=True))
                        concept_file.write(values.tobytes())
                saved_annotations = os.path.join(staging, "annotations.pt")
                torch.save(annotations, saved_annotations)
                os.replace(ids, id_path)
                os.replace(concepts, concept_path)
                os.replace(saved_annotations, annotation_path)

    def load_raw(self):
        """Open the artefacts already prepared by build(), as in CUB."""
        self.maybe_build()
        _, id_path, concept_path, annotation_path = self.processed_paths
        annotations = torch.load(annotation_path, weights_only=False)
        count = os.path.getsize(id_path) // 8
        inputs = torch.from_file(id_path, shared=False, size=count, dtype=torch.int64)
        concepts = torch.from_file(concept_path, shared=False,
                                   size=count * len(annotations.labels), dtype=torch.float32)
        return inputs, concepts.reshape(count, -1), annotations, None

    def _db(self):
        if self._connection is None or self._connection_pid != os.getpid():
            self.close()
            path = Path(self.processed_paths[0]).resolve().as_uri() + "?mode=ro"
            self._connection = sqlite3.connect(path, uri=True)
            self._connection_pid = os.getpid()
        return self._connection

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __getstate__(self):
        """Serialize the dataset without its process-local SQLite connection."""
        state = self.__dict__.copy()
        state["_connection"] = None
        state["_connection_pid"] = None
        # AnnotatedTensor itself is not pickle-safe in the shared library.
        for name in ("native_concepts", "concepts"):
            values = state[name]
            if values is not None:
                state[name] = _BatchTensor(values.tensor, values.annotations, values.axis)
        state["generated_concepts"] = {
            name: _BatchTensor(values.tensor, values.annotations, values.axis)
            for name, values in self.generated_concepts.items()
        }
        return state

    def __getitem__(self, item):
        sample = super().__getitem__(item)
        if not self.embs_precomputed:
            sample["inputs"]["x"] = self._db().execute(
                "SELECT sequence FROM proteins WHERE id = ?", (int(sample["inputs"]["x"]),)
            ).fetchone()[0]
        return sample

    def collate(self, samples):
        # Reuse the library's collation. Only add pickle support to these batch
        # tensors, so workers work without changing AnnotatedTensor itself.
        batch = super().collate(samples)
        for group in (batch["concepts"], batch["concepts"]["generated"]):
            for name, values in group.items():
                if isinstance(values, AnnotatedTensor):
                    group[name] = _BatchTensor(values.tensor, values.annotations, values.axis)
        return batch
