#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle


def tiller_square_api(arg):
    """
    :param arg: arg could be a n by 6 ndarray with each column representing 'EVI', 'NDVI', 'GDVI', 'OSAVI', 'PSRI' and 'NDRE';
    :return: tiller squares
    """
    assert isinstance(arg, np.ndarray) or isinstance(arg, pd.DataFrame), "numpy.ndarray or pandas.dataframe is required for input."
    assert arg.size%6 == 0, "Six variables are required ('EVI', 'NDVI', 'GDVI', 'OSAVI', 'PSRI' and 'NDRE')"

    filename = 'finalized_model.sav'
    rf = pickle.load(open(filename, 'rb'))
    tiller_square_predict = rf.predict(arg)
    return tiller_square_predict

if __name__ == '__main__':
    input = pd.read_csv(r'combined.csv')
    tiller_square_api(input[['evi', 'ndvi', 'gdvi', 'osavi', 'psri', 'ndre']])


# import urllib.request as urllib2
# import json
#
# def main():
#     rawdata = None
#     with open("combined.csv", "r") as f:
#         rawdata = f.read()
#     # print(rawdata)
#     data = {}
#     data['values'] = rawdata
#     body = str.encode(json.dumps(data))
#     url = 'http://digitalag.taegon.kr:5000/tiller_square'
#     headers = {'Content-Type': 'application/json',}
#     req = urllib2.Request(url, body, headers)
#     try:
#         response = urllib2.urlopen(req)
#         result = response.read()
#         result_parse = eval(str(result, encoding="utf-8"))
#         print(result_parse)
#         print(len(result_parse))
#     except urllib2.HTTPError.error:
#         print("The request failed with status code: " + str(error.code))
#
#
# if __name__ == "__main__":
#     main()



