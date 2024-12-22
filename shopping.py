import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])

    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):

    import calendar

    #month name search by abbrevation
    month_search =  {name: index for index, name in enumerate(calendar.month_abbr) if name}

    #month name search by full name
    full_month_names = {name: index for index, name in enumerate(calendar.month_name) if name}
    month_search.update(full_month_names)

    evidence = []
    labels = []

    with open (filename, mode = 'r') as file:
        csvFile = csv.DictReader(file)
        for row in csvFile:
            evidence.append([
                int (row["Administrative"]),
                float(row["Administrative_Duration"]),
                int (row["Informational"]),
                float (row["Informational_Duration"]),
                int (row["ProductRelated"]),
                float (row["ProductRelated_Duration"]),
                float (row["BounceRates"]),
                float (row["ExitRates"]),
                float (row["PageValues"]),
                float (row["SpecialDay"]),
                month_search[row["Month"]],
                int (row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0
            ])

            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    return evidence, labels


def train_model(evidence, labels):

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model
    
    
def evaluate(labels, predictions):

    true_pos = sum(1 for actual, pred in zip(labels, predictions) if actual == pred == 1)
    true_neg = sum(1 for actual, pred in zip(labels, predictions) if actual == pred == 0)
    tot_pos = sum(1 for label in labels if label == 1)
    tot_neg = sum(1 for label in labels if label == 0)

    sensitivity = true_pos / tot_pos if tot_pos else 0
    specificity = true_neg / tot_neg if tot_neg else 0
    return sensitivity, specificity


if __name__ == "__main__":
    main()
