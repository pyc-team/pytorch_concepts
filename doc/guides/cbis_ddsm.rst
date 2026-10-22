CBIS-DDSM
=========

:class:`~torch_concepts.data.CBISDDSMDataset` exposes the Curated Breast Imaging
Subset of DDSM as abnormality/view observations. Its primary reference is
`Lee et al., A curated mammography data set for use in computer-aided detection
and diagnosis research, Scientific Data 4, 170177 (2017)
<https://doi.org/10.1038/sdata.2017.177>`_. The images originate from scanned-film
mammography. The authors decompressed DDSM, processed scanner intensities into
optical density, and distributed 16-bit grayscale images as DICOM. This loader
does not repeat that historical processing.

Acquisition and layout
----------------------

Install the data extras (including ``pydicom`` and ``tables``). The dataset first
tries to download the selected metadata CSV and missing image series from TCIA's
public NBIA API. Only series referenced by the requested abnormality, split, and
image product are requested; existing local images are reused. NBIA transfers
whole series, which can contain both a crop and a mask. Only requested instances
are retained. Large selections can require substantial time and disk space.

If a transfer fails, or downloaded instances cannot be matched unambiguously,
the dataset raises an error with the cause and manual acquisition instructions.
Completed files remain available for the next attempt; incomplete transfers and
temporary ZIP archives are removed. Network requests use a 60-second timeout.
There is no interactive terminal prompt, so notebooks and training jobs receive
the same actionable error.

For manual acquisition, use the NBIA Data Retriever from the `official TCIA
collection <https://www.cancerimagingarchive.net/collection/cbis-ddsm/>`_. Download
the full mammogram and ROI/crop products needed for your selection, and place
the metadata CSVs and images as follows:

.. code-block:: text

   root/
     raw/
       mass_case_description_train_set.csv
       mass_case_description_test_set.csv
       calc_case_description_train_set.csv
       calc_case_description_test_set.csv
     images/                         # arbitrary extraction prefix
       .../StudyUID/SeriesUID/000000.dcm
       .../StudyUID/SeriesUID/000001.dcm
     processed/                      # created by PyC; no pixel copies
       mass/train/cropped/
         concepts.h5
         annotations.pt
         metadata.h5

Only the selected CSV and image product are required. DICOMs may be anywhere
below ``root``. The resolver indexes the final three path components
(``StudyUID/SeriesUID/filename.dcm``), ignoring extraction and CSV prefixes.
Whitespace surrounding references, including source trailing newlines, is
ignored for resolution while the original cell is preserved in metadata.
Exactly one match is required, including when a file exists at the exact CSV
path: duplicate local copies are ambiguous. Zero matches and multiple matches
raise errors identifying the reference, observation, and requested column.
Exports that rename UID directories or instance filenames must restore these
suffixes using an independently verified mapping. The loader does not guess an
instance from directory order, a numeric filename, or image dimensions.

The automatic downloader checks StudyInstanceUID and SeriesInstanceUID in each
selected DICOM header before retaining it. An API member must match the CSV
filename exactly. If a series contains only one DICOM and only one filename is
requested, that unique instance can be saved under the CSV name. For multi-image
series with incompatible filenames, acquisition stops with manual instructions.
In a live check on 16 October 2026, the API returned a crop/mask series as
``00000001.dcm`` and ``00000002.dcm``, while the CSV used ``000000.dcm`` and
``000001.dcm``. Both instances had InstanceNumber 1, so that field could not
establish a unique filename mapping either. The downloader does not rename by
position, swap columns, or infer the intended file from image appearance.
See the `official NBIA download API documentation
<https://wiki.cancerimagingarchive.net/display/Public/NBIA+Search+REST+API+Guide>`_.

Caches are separate for each abnormality/split/image selection. Images remain
DICOM sources. Existing caches are reused deterministically and their selected
paths are checked for existence. After editing CSVs or relocating images within
the root, delete the affected processed directory to rebuild. Moving the entire
root preserves the relative cached paths. Native image sizes may differ;
``n_features`` and ``shape`` describe the first sample's shape. Supply an
appropriate transform or collator when batching variable-sized images.

API and sample identity
-----------------------

For a complete loading and CLIP concept-generation example, see
:doc:`/auto_examples/data/07_cbis_ddsm`.

.. code-block:: python

   from torch_concepts.data import CBISDDSMDataset

   dataset = CBISDDSMDataset(
       root="/path/to/cbis_ddsm",
       abnormality_type="mass",   # or "calcification"
       split="train",            # or "test"
       image_type="cropped",     # or "full", "mask"
       transform=None,
       concept_subset=None,
   )
   x = dataset[0]["inputs"]["x"]
   source = dataset.metadata.iloc[0]

The official CSV's membership defines the split; no random split is generated.
The 2017 descriptor explains that mass and calcification were split separately,
with approximately 20% of cases assigned to testing, stratified by assessment.

One sample is one CSV observation, describing an abnormality on one breast/view.
A clinical abnormality can appear on CC, MLO, or both. Multiple abnormalities
can share a mammogram. Every observation is retained, even repeated rows, so
``image_type="full"`` can return identical pixels with different annotations.
``patient_id`` comes from the CSV rather than the DICOM PatientID.

``sample_id`` joins abnormality type, official split, CSV patient ID, laterality,
view, abnormality ID, and a zero-based occurrence number for repeated identities.
``source_row`` is the zero-based CSV observation position. Use patient ID for
patient grouping, patient/laterality for breast grouping, and
patient/laterality/abnormality ID/type for cross-view abnormality grouping;
the original full-image reference is available for mammogram grouping.
Abnormality IDs are not globally unique. No ``all`` selection is provided, to
keep structurally inapplicable descriptor families off each concept axis.

Image products and transforms
-----------------------------

The selected source is explicit:

.. list-table::
   :header-rows: 1

   * - image_type
     - Source CSV column
     - Product
   * - full
     - image file path
     - Full mammogram
   * - cropped
     - cropped image file path
     - Distributed crop from the abnormality ROI's bounding rectangle
   * - mask
     - ROI mask file path
     - Distributed binary ROI mask at mammogram resolution

Crops are never reconstructed. The original descriptor reports improved
automated segmentation for masses only; calcification masks have different
provenance. The loader uses both distributed products as supplied.

``pydicom.pixel_array`` decodes a single monochrome frame. The result is a
``float32`` tensor of shape ``(1, H, W)``. All 16-bit integers are exactly
representable. MONOCHROME2 stored samples are unchanged. For MONOCHROME1 the
loader uses ``low + high - pixel``, with endpoints from ``BitsStored`` and
``PixelRepresentation``, to express increasing brightness consistently. This
is a fixed affine inversion, not per-image normalization. No modality/rescale
or VOI/windowing LUT is applied. Mask samples are not thresholded or divided by
255. Unsupported color or multiframe inputs raise errors.

``transform`` receives this native grayscale tensor. Resizing, padding,
min-max normalization, RGB replication, and backbone normalization belong to
that callable. Transforms are skipped when inputs are precomputed embeddings,
consistent with the other image datasets.

Annotations and source metadata
-------------------------------

Columns are normalized centrally by trimming, lowercasing, and replacing
whitespace with underscores; cell strings remain unchanged. All source columns,
including all three original path references, remain in ``metadata``, together
with ``split``, ``source_csv``, ``source_row``, ``sample_id``, and
``resolved_image_path`` for the selected image product. Metadata follows PyC
row subsetting in the same order as images and annotations.

The first four concepts have this fixed order and categorical state order:

.. list-table::
   :header-rows: 1

   * - Concept
     - Ordered states
   * - breast_density
     - 1, 2, 3, 4, 0
   * - assessment
     - 0, 1, 2, 3, 4, 5
   * - subtlety
     - 0, 1, 2, 3, 4, 5
   * - pathology
     - BENIGN, BENIGN_WITHOUT_CALLBACK, MALIGNANT

PyC stores categorical values as zero-based state indices, not continuous
measurements. Density categories 1–4 describe increasing breast density.
The exceptional source density ``0`` is retained as a fifth distinct state
(index 4), with no inferred meaning. Density is never normalized. Subtlety
includes ``0`` because the released CSV contains it despite the paper's stated
1–5 range. Pathology has three states; benign and benign-without-callback are
never merged. Blank categorical cells produce NaN, and unexpected states
raise errors rather than inventing a mapping.

The applicable descriptor families follow these four categorical concepts.
Their atomic binary labels are ``family.SOURCE_TOKEN``, with this fixed order:

* ``mass_shape``: N/A, ARCHITECTURAL_DISTORTION, ASYMMETRIC_BREAST_TISSUE,
  FOCAL_ASYMMETRIC_DENSITY, IRREGULAR, LOBULATED, LYMPH_NODE, OVAL, ROUND.
* ``mass_margin``: N/A, CIRCUMSCRIBED, ILL_DEFINED, MICROLOBULATED, OBSCURED,
  SPICULATED.
* ``calcification_type``: N/A, AMORPHOUS, COARSE, DYSTROPHIC, EGGSHELL,
  FINE_LINEAR_BRANCHING, LARGE_RODLIKE, LUCENT_CENTER, LUCENT_CENTERED,
  MILK_OF_CALCIUM, PLEOMORPHIC, PUNCTATE, ROUND_AND_REGULAR, SKIN, VASCULAR.
* ``calcification_distribution``: N/A, CLUSTERED, DIFFUSELY_SCATTERED, LINEAR,
  REGIONAL, SEGMENTAL.

Hyphens separate atomic descriptors. ``OVAL-LOBULATED`` sets both indicators
to 1; absent descriptors in a recorded, applicable family are 0. Repeated
tokens such as ``REGIONAL-REGIONAL`` indicate presence once; the original
string is retained. ``LUCENT_CENTER`` and ``LUCENT_CENTERED`` stay separate.
There is no spelling correction or clinical canonicalization. These vocabularies
were checked against all four official CSVs and the `TFDS reference vocabularies
<https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image_classification>`_.

Literal ``N/A`` is not interpreted as a pandas missing value. It sets the
family's ``N/A`` indicator to 1 and leaves medical descriptor indicators NaN.
A genuinely blank family leaves every indicator NaN, including ``N/A``.
A recorded applicable descriptor sets ``N/A`` to 0. Structurally inapplicable
families are absent from the axis. Nothing is imputed. Unknown tokens or a
compound containing both N/A and medical descriptors raise errors.

The native axes contain 19 mass concepts and 25 calcification concepts. Binary
concepts use PyC's one-logit ``["present"]`` state schema and values 0/1/NaN.
The ``density`` group aliases ``breast_density``, descriptor family groups
select their indicators, and ``radiological`` selects everything except
pathology. ``pathology``, ``assessment``, and ``subtlety`` are directly selectable
concept names. ``concept_subset`` accepts names and groups in requested order.
Identifiers, paths, laterality, view, abnormality type, and split remain metadata.

Known source discrepancies
--------------------------

An audit of the official TCIA CSVs on 11 October 2026 found:

.. list-table:: CSV identity levels
   :header-rows: 1

   * - Selection
     - Observations
     - Patients
     - Breast/view identities
     - Cross-view abnormality identities
     - Repeated full-image references
   * - Mass train
     - 1318
     - 691
     - 1231
     - 782
     - 87
   * - Mass test
     - 378
     - 201
     - 361
     - 223
     - 17
   * - Calcification train
     - 1546
     - 602
     - 1227
     - 856
     - 319
   * - Calcification test
     - 326
     - 151
     - 284
     - 189
     - 42

Pathology observation counts in BENIGN/BENIGN_WITHOUT_CALLBACK/MALIGNANT order
are 577/104/637 (mass train), 194/37/147 (mass test), 528/474/544
(calcification train), and 130/67/129 (calcification test). Density 1/2/3/4/0
observation counts are respectively 287/585/337/109/0, 50/172/112/44/0,
175/501/539/331/0, and 24/115/115/70/2. No descriptor cells were blank.
N/A counts for shape/margin were 4/43 in mass train and 0/17 in mass test;
type/distribution counts were 20/376 in calcification train and 4/63 in test.

The paper reports 753 calcification cases and 891 mass cases, with Table 3
abnormality totals of 856/189 for calcification train/test and 748/222 for mass.
The CSV cross-view abnormality counts agree for calcification but exceed the
mass totals by 34 (train) and 1 (test). The mass test CSV has 201 patients,
compared with 200 cases in the paper's table; patients and cases are different
identity levels, so these figures must not be equated. Inspection found no
duplicate observation identities and no within-type patient overlap between
train and test. Three cross-view abnormality identities in each abnormality
type contain differing pathology strings. The CSVs do not provide a mapping
to the paper's historical case identities that resolves these differences.
All released observations and their per-view annotations are retained.

No real DICOM collection was available for that audit, so image resolution and
image content could not be verified. A metadata-only audit will report every
image reference unresolved; this does not demonstrate a defect in the release.
The `TFDS loader
<https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/cbis_ddsm.py>`_
also documents possible crop/mask column reversals and incompatible dimensions.
This implementation maps the official columns directly and does not swap images
or discard rows based on dimensions. Content ambiguities require source review;
the selected column alone cannot guarantee that a mislabeled source file is
the intended product. The authors' `DDSM curation tooling
<https://github.com/fjeg/ddsm_tools>`_ provides additional provenance.

Downstream experiments
----------------------

Paper-specific benign/malignant binarization, density remapping, selected
18- or 14-dimensional concept projections, lesion normalization, square padding,
224-by-224 resizing, RGB replication, ImageNet normalization, patient-level
cross-validation, and annotation-fraction sampling are left to transforms and
examples. No architecture, prior initialization, or experimental loss is part
of this dataset.
