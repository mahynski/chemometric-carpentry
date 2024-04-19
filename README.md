Project Name
---
1. Insert description here.
2. Rename repo with a 4 digit year-of-completion prefix, e.g., "2022-"; this can be updated later. Convention is to use hyphens between words and all lower case.
3. Create a [conda](https://www.anaconda.com/) environment for this project.  First modify `conda-env.yml` to include the relevant repositories and dependencies needed; also give the environment a good name (e.g., similar or same as this repo). Then create the environment (see below).

Installation
---
Set up the conda environment for this project.
```code
$ conda env create -f conda-env.yml
$ conda activate PROJECT-NAME
$ python -m ipykernel install --user --name=PROJECT-NAME
```

It is also useful to export the entire conda environment for posterity.
```code
$ conda env export > environment.yml
```

Contributors
---
Update the CITATION.cff file to enable appropriate citations.  

Versioning
---
* Use the [public-template](https://github.com/mahynski/public-template) to create a fresh repo to release the code and details after a manuscript is published, tag the release, then use zenodo to capture changes to future changes/releases made to that repo. This serves as the primary **public facing** repo.
* In addition, create a "published" branch on this repo to correspond to when the associated paper is published. This is retained as the primary **private** repo where future work can be tested. Subsequent branches, such as "revision-YYYY-MM-DD" can be created later and similarly reflected in the public-template version if revisions are necessary. 

Associated Publications
---
[LINK TO MANUSCIPT]()
