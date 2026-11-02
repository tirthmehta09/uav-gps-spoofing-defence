"""
The repo's conference-template-a4.docx is a STRICT Open XML file (namespaces under
http://purl.oclc.org/ooxml/...), which python-docx cannot open. This rewrites those
namespace/relationship URIs to their TRANSITIONAL equivalents
(http://schemas.openxmlformats.org/.../2006/...) in every XML part, preserving all
IEEE styles/layout, and writes conference-template-a4-transitional.docx.
"""
import os
import re
import shutil
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE_DIR, "conference-template-a4.docx")
DST = os.path.join(BASE_DIR, "conference-template-a4-transitional.docx")

# Ordered so longer/more-specific relationship suffixes are fixed before the generic prefix.
REPLACEMENTS = [
    # schema namespaces for the metadata parts (these use /2006/ and hyphenated local names)
    ("http://purl.oclc.org/ooxml/officeDocument/extendedProperties",
     "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"),
    ("http://purl.oclc.org/ooxml/officeDocument/customProperties",
     "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"),
    ("http://purl.oclc.org/ooxml/officeDocument/docPropsVTypes",
     "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"),
    ("http://purl.oclc.org/ooxml/officeDocument/bibliography",
     "http://schemas.openxmlformats.org/officeDocument/2006/bibliography"),
    # relationship types
    ("http://purl.oclc.org/ooxml/officeDocument/relationships/extendedProperties",
     "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties"),
    ("http://purl.oclc.org/ooxml/officeDocument/relationships",
     "http://schemas.openxmlformats.org/officeDocument/2006/relationships"),
    ("http://purl.oclc.org/ooxml/wordprocessingml/main",
     "http://schemas.openxmlformats.org/wordprocessingml/2006/main"),
    ("http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing",
     "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"),
    ("http://purl.oclc.org/ooxml/drawingml/picture",
     "http://schemas.openxmlformats.org/drawingml/2006/picture"),
    ("http://purl.oclc.org/ooxml/drawingml/main",
     "http://schemas.openxmlformats.org/drawingml/2006/main"),
    ("http://purl.oclc.org/ooxml/officeDocument/math",
     "http://schemas.openxmlformats.org/officeDocument/2006/math"),
    ("http://purl.oclc.org/ooxml/schemaLibrary/main",
     "http://schemas.openxmlformats.org/schemaLibrary/2006/main"),
    # any remaining ooxml namespace -> generic transitional root (safety net)
    ("http://purl.oclc.org/ooxml/", "http://schemas.openxmlformats.org/"),
]


def convert(text):
    for a, b in REPLACEMENTS:
        text = text.replace(a, b)
    return text


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"missing {SRC}")
    zin = zipfile.ZipFile(SRC, "r")
    if os.path.exists(DST):
        os.remove(DST)
    zout = zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED)
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename.endswith((".xml", ".rels")):
            data = convert(data.decode("utf-8")).encode("utf-8")
        zout.writestr(item, data)
    zin.close(); zout.close()
    print(f"Wrote {DST}")

    # verify python-docx can open it
    from docx import Document
    d = Document(DST)
    print(f"python-docx OPENED it OK. paragraphs={len(d.paragraphs)} tables={len(d.tables)} "
          f"styles={len(d.styles)}")


if __name__ == "__main__":
    main()
