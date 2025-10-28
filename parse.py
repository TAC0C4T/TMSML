import csv

def parse(file: str, columns: list[str]) -> list[list[float]]:
    with open(file, newline='') as f:
        reader = csv.DictReader(f)
        result = []
        for row in reader:
            result.append([float(row[col]) for col in columns])
    return result
