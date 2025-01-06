# BloodSeg-UNETR

All model can be run using pretrained ViTMAE Models.

To run or train a model, the correct ViTMAE configs that are located in logs/vitmae/info must be copied to the '
vitmaeconfig' config dictionary along with model path of model.pth

## Models

UNETR2 (2x2): Models must be run/trained using a pretrained ViTMAE model with 2x2 patch size

UNETR (4x4): Models must be run/trained using a pretrained ViTMAE model with 4x4 patch size

UNETR8 (8x8): Models must be run/trained using a pretrained ViTMAE model with 8x8 patch size

## Data-Extraction

For train, valid and test data, the data extraction is located in notebooks/datasets. Labels and images are zipped into
a npz files, which is then unzipped when loaded in 'unetr.py'.

## View Result

To view results and metrics of a trained UNETR, copy 'view_results.ipynb'. Make sure to change the vitconfig dictionary
and unetconfig dictionary as well as the model path which can all be found in info.md 
