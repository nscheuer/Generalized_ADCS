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

## Generating Documentation
Write a Sphinx+LaTeX documentation for this function, including:
- r""" strings
- Full math explanation with equations
- When referencing other classes or methods, use the full link ~ADCS.etc
- Use :param: and :return:, include the full param and return type
- When using or creating tables, follow reST rules
Write only the function header and the documentation.