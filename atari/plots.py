#!/usr/bin/env python3

import os
from glob import glob
import csv
from matplotlib import pyplot as plt
import filecmp


PATH = "output"
EXT = "*.csv"

def getCSVFiles():
    all_csv_files = [file
        for path, subdir, files in os.walk(PATH)
        for file in glob(os.path.join(path, EXT))]
    return all_csv_files


def main():
    csvFiles = getCSVFiles()
    for c in csvFiles:
        x = []
        y = []
        with open(c) as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                if row[0].isdigit():
                    x.append(float(row[0]))
                    y.append(float(row[1]))
        fileName = list(os.path.splitext(c))     # returns tuple (file, ext)
        fileName[0] = fileName[0] + ".png"
        plotName = os.path.join(fileName[0])
<<<<<<< HEAD
        print("creating: ", plotName)       
        plt.scatter(x, y)
        plt.xlabel(headers[0])
        plt.ylabel(headers[1])
        plt.title(fileName[0])
        plt.savefig(plotName)
        plt.clf()
        plt.close()
        
=======
               
        plt.scatter(x, y)
        plt.savefig(plotName)
        plt.clf()
        plt.close()

>>>>>>> 4737c4178447e026f976ade827c6c8f0e4edfd11

if __name__ == "__main__":
    main()
