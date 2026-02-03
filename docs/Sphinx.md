# Sphinx Instructions
In order to regenerate the Sphinx documentation, enter the `docs` folder and rebuild the project. Ensure that the virtual environment is activated.
```bash
source venv/bin/activate
cd docs/
make clean
sphinx-build -W -b html source build
```

In order to register new Python modules (generate new .rst files):
```bash
cd docs/
find source -maxdepth 1 -type f -name "*.rst" ! -name "index.rst" -delete
sphinx-apidoc -f -e --maxdepth 1 -o source/ ../ADCS/
```

## Viewing Documentation
Open the [index.html](../docs/build/html/index.html) file in a browser or IDE.

## Generating Documentation
Write a Sphinx+LaTeX documentation for these classes and functions, including:
- r""" strings
- Full math explanation with equations
- When referencing other classes or methods, use the full link :class:`~ADCS.etc`
- STRICTLY use :param:, :type:, :return:, :rtype:
- When using or creating tables, follow reST rules, means no ** allowed
- Every docstring field list must end with a blank line.
Write only the function header and the documentation.

Review your results and compare it with the rules above. If one of the rules is broken, notice and fix it.