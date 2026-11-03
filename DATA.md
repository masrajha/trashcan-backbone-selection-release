# Dataset access and split

Download the original TrashCan dataset from the University of Minnesota IRV Lab resource page: https://irvlab.cs.umn.edu/resources/trashcan . Follow the original dataset terms. This release does not redistribute images or annotations.

For the published Phase 3 experiments, place the **material** release under `material_version/` in this repository root. The training code expects `material_version/instances_train_trashcan.json`, `material_version/instances_val_trashcan.json`, and the image paths referenced by those COCO files. Verify the dataset version and layout before running. The official test set was not used for backbone selection.

The exact frozen internal train/validation membership is in `splits/internal_split_manifest_final.json` (4,800 train images; 1,211 validation images). Its `source_annotation_sha256` identifies the expected training annotation file. The same manifest is copied to `Riset/output/research_phase1/internal_split_manifest.json`, the default path used by `utils/experiment.py`. Group aliases and image IDs remain intact.

Do not create a new random split for comparison with the reported results.
