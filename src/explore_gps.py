"""
Quick structural exploration of the GPS spoofing xlsx using openpyxl read-only mode.
Avoids loading the whole 106MB file into pandas at once.
"""
import os
import openpyxl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(BASE_DIR, "dataset", "GPS_Spoofing_UAV.xlsx")

def explore():
    print(f"Opening {PATH} in read-only mode...")
    wb = openpyxl.load_workbook(PATH, read_only=True, data_only=True)
    print(f"Sheet names: {wb.sheetnames}")

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        print(f"\n=== Sheet: {sheet_name} ===")
        print(f"Dimensions (reported): max_row={ws.max_row}, max_col={ws.max_column}")

        rows_iter = ws.iter_rows(values_only=True)
        header = next(rows_iter, None)
        print(f"Header ({len(header) if header else 0} cols): {header}")

        # Print first 5 sample data rows
        print("Sample rows:")
        for i, row in enumerate(rows_iter):
            print(f"  {row}")
            if i >= 4:
                break

        # Count rows + tally last column (likely label) values
        # (stream through; only for the first/main sheet to save time)
        label_counts = {}
        total = 0
        # Re-open iterator for full count
        for row in ws.iter_rows(values_only=True):
            if total == 0:
                total += 1
                continue  # skip header
            total += 1
            if row and len(row) > 0:
                label = row[-1]
                label_counts[label] = label_counts.get(label, 0) + 1
        print(f"Total rows (incl header): {total}")
        print(f"Last-column value counts (top 20): {dict(list(sorted(label_counts.items(), key=lambda x: -x[1]))[:20])}")

    wb.close()

if __name__ == "__main__":
    explore()
