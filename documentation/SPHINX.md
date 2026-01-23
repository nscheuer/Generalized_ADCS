# Sphinx Instructions
In order to regenerate the Sphinx documentation, enter the `docs` folder and rebuild the project. Ensure that the virtual environment is activated.
```bash
source venv/bin/activate
cd docs/
make clean
make html
```

In order to register new Python modules (generate new .rst files):
```bash
cd docs/
sphinx-apidoc -f -e --maxdepth 1 -o source/ ../ADCS/
```

## Viewing Documentation
Open the [index.html](../docs/build/html/index.html) file in a browser or IDE.