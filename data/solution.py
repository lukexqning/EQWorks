import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('pandas')

import pandas as pd
from pyspark import SparkContext, SparkConf

if __name__ == "__main__":
    conf = SparkConf().setAppName("EQWorksLukeNing")
    sc = SparkContext(conf=conf)

    data_df = pd.read_csv("/tmp/data/DataSample.csv")
    data_df.rename(columns=lambda x: x.strip(), inplace=True)
    print('Original length: ', end="")
    print(len(data_df))

    ## Part 1

    data_df.drop_duplicates(["TimeSt", "Latitude", "Longitude"], keep="last", inplace=True)

    print('\nPart 1')
    print('Deduped length: ', end="")
    print(len(data_df))

    ## Part 2

    poi_df = pd.read_csv("/tmp/data/POIList.csv")
    poi_df.rename(columns=lambda x: x.strip(), inplace=True)
    poi_map = {}

    for _, row in poi_df.iterrows():
        poi_lst = poi_map.get((row["Latitude"], row["Longitude"]))
        poi_map[(row["Latitude"], row["Longitude"])] = (poi_lst or []) + [row["POIID"]]

    def calc_dist(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    def closest_poi(row, mapping):
        distances = {k: calc_dist([row["Latitude"], row["Longitude"]], k) for k in mapping.keys()}
        place = min(distances, key=distances.get)
        return mapping[place]

    data_df['ClosestPOI'] = data_df.apply(lambda row: closest_poi(row, poi_map), axis=1).astype(str)

    print('\nPart 2')
    print(data_df.groupby(['ClosestPOI']).agg(['count']))

    ## Part 3

    def poi_dist(row, mapping):
        return min([calc_dist([row["Latitude"], row["Longitude"]], k) for k in mapping.keys()])

    data_df['ClosestDist'] = data_df.apply(lambda row: poi_dist(row, poi_map), axis=1)

    # 1)

    summary = data_df[['ClosestPOI', 'ClosestDist']].groupby(['ClosestPOI']).agg(['mean', 'std'])
    print('\nPart 3, 1)')
    print(summary)

    # 2)

    def poi_area(row):
        PI = 3.141592653589793
        tot = row['count']
        area = PI*(row['max']**2)
        return tot/area

    poi_range = data_df[['ClosestPOI', 'ClosestDist']].groupby(['ClosestPOI']).agg(['max', 'count'])
    poi_range.columns = poi_range.columns.droplevel()
    poi_range['density'] = poi_range.apply(lambda row: poi_area(row), axis=1)
    poi_range.drop(['count'], axis=1, inplace=True)
    poi_range.rename(columns={'max': 'radius'}, inplace=True)

    print('\nPart 3, 2)')
    print(poi_range)

    ## Part 4 b)

    # Implementation
    ## shortest_path(starts, goal, order) produces the least amount of
    ##     tasks required to accomplish goal from start, using the
    ##     dependencies found in order. If a path doesn't exist, False
    ##     is returned
    ## shortest_path: (setof Str) Str (dictof Str (listof Str)) -> (anyof (listof Str) False)
    def shortest_path(starts, goal, graph):
        for start in starts:
            graph[start] = set()
        path = [goal]
        queue = [goal]
        while queue:
            v = queue.pop()
            for task in (graph.get(v) or set()) - set(path):
                path = [task] + path
                queue.append(task)
        return path

    # Data Cleaning
    starts = set()
    goal = ''

    question = open("/tmp/data/question.txt", "r")
    starts = question.readline().strip().split(':')[1]
    starts = {x.strip() for x in starts.split(',')}

    goal = question.readline().strip().split(':')[1].strip()

    graph = {}

    relations = open("/tmp/data/relations.txt", "r")
    for line in relations:
        vals = line.strip().split('->')
        graph[vals[1]] = (graph.get(vals[1]) or set()).union({vals[0]})

    print('\nPart 4b')
    # Example Result
    print('\nExample Result')
    print(shortest_path(starts, goal, graph))

    # Test from repo
    graph = {
        'C': {'A', 'B'},
        'E': {'C'},
        'F': {'E'}}

    print('\nExamples from Repo, Start=A')
    print(list(shortest_path({'A'}, 'F', graph)))
    print('\nExamples from Repo, Start=A,C')
    print(list(shortest_path({'A', 'C'}, 'F', graph)))

    sc.stop()
