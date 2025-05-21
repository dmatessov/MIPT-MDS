import csv

with open('./data_cats/cats_categories.csv', newline='',) as csvfile:

    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    header = next(spamreader)
    for row in spamreader:
        values = str(row[0]).split(',')
        filename = values[1]
        cat = int(values[2])
        print(filename, cat)