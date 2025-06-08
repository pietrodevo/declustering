# -*- coding: utf-8 -*-

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""MANIFEST"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

"""

author: pietro devò
e-mail: pietro.devo@dicea.unipandas.it

            .==,_
           .===,_`^
         .====,_ ` ^      .====,__
   ---     .==-,`~. ^           `:`.__,
    ---      `~~=-.  ^           /^^^
      ---       `~~=. ^         /
                   `~. ^       /
                     ~. ^____./
                       `.=====)
                    ___.--~~~--.__
          ___|.--~~~              ~~~---.._|/
          ~~~"                             /

"""

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""LIBRARIES"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

import numpy
import pandas

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""FUNCTIONS"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def declustering(
    dataframe,
    frequency="median",
    column=None,
    formatter=True,
    threshold=0.99,
    window="06d",
    sed=None,
    sm=None,
    output=None,
):
    """

    Ariadna Martín, Thomas Wahl, Alejandra R. Enriquez, Robert Jane

    Storm Surge Time Series De-Clustering Using Correlation Analysis

    dataframe -> input dataframe;
    frequency -> time frequency value/infer;
    column    -> data column;
    formatter -> time label formatter;
    threshold -> height threshold value for surges identification;
    window    -> sampling window for sed/sm estimation;
    sed       -> optional standard event duration;
    sm        -> optional time steps separation margin;
    output    -> optional output filter.

    """

    # copying dataframe
    dataframe = dataframe.copy()

    if type(frequency).__name__ != "timedelta":
        frequency = pandas.Timedelta(frequency)

    if type(dataframe).__name__ == "Series":
        dataframe = dataframe.to_frame()

    if column is None:
        column = dataframe.columns[0]

    # getting index name
    index = dataframe.index.name

    if threshold is None:
        threshold = numpy.quantile(
            dataframe.loc[dataframe.loc[:, column] >= 0, column]
        )
    else:
        threshold = 0

    # events over threshold
    eot = dataframe.loc[
        dataframe.loc[:, column] >= threshold, column
    ].reset_index()

    # parameters array
    parameters = numpy.array([sed, sm], dtype=float)

    if (numpy.isnan(parameters)).any():

        # computing time lag
        lag = pandas.Timedelta(window) / 2

        # initialize dictionary
        dictionary = {
            i: {
                "independent": None,
                "parent": None,
                "maxima": j[column],
                "datetime": j[index],
            }
            for i, j in eot.iterrows()
        }

        # initialize check
        check = numpy.full(eot.shape[0], True)

        # initialize indexer
        indexer = []

        while check.any():

            # indexing
            i = eot.loc[check, column].idxmax()

            # updating dictionary
            dictionary[i]["independent"] = True
            dictionary[i]["start"] = dictionary[i]["datetime"] - lag
            dictionary[i]["end"] = dictionary[i]["datetime"] + lag
            dictionary[i]["data"] = dataframe.loc[
                dictionary[i]["start"] : dictionary[i]["end"], column
            ].to_numpy()
            dictionary[i]["nan"] = numpy.isnan(dictionary[i]["data"]).any()
            dictionary[i]["inf"] = numpy.isinf(dictionary[i]["data"]).any()
            dictionary[i]["events"] = eot.index[
                (
                    (eot.loc[:, index] >= dictionary[i]["start"])
                    & (eot.loc[:, index] <= dictionary[i]["end"])
                    & (eot.loc[:, index] != dictionary[i]["datetime"])
                )
            ]

            for e in dictionary[i]["events"]:
                dictionary[e]["independent"] = False
                dictionary[e]["parent"] = i

            # unflagging events
            check[dictionary[i]["events"].union([i])] = False

            # appending indexer
            indexer.append(i)

        # correlation matrix
        correlation = (
            pandas.DataFrame([dictionary[i]["data"] for i in indexer])
            .fillna(0)
            .corr(numeric_only=True)
        )

        # computing statistics
        M = correlation.mean(axis=1)
        S = correlation.std(axis=1)
        C = M.size / 2

        # standard event duration
        parameters[0] = abs(M.argmax() + 1 - C) * 2

        # time steps separation margin
        parameters[1] = abs(parameters[0] / 2 - abs((M + S).argmax() + 1 - C))

    # computing time lag
    lag = pandas.Timedelta(hours=parameters[0]) / 2

    # initialize dictionary
    dictionary = {
        i: {
            "independent": None,
            "parent": None,
            "maxima": j[column],
            "datetime": j[index],
        }
        for i, j in eot.iterrows()
    }

    # initialize check
    check = numpy.full(eot.shape[0], True)

    # initialize indexer
    indexer = []

    while check.any():

        # indexing
        i = eot.loc[check, column].idxmax()

        # updating dictionary
        dictionary[i]["independent"] = True
        dictionary[i]["start"] = dictionary[i]["datetime"] - lag
        dictionary[i]["end"] = dictionary[i]["datetime"] + lag
        dictionary[i]["data"] = dataframe.loc[
            dictionary[i]["start"] : dictionary[i]["end"], column
        ].to_numpy()
        dictionary[i]["nan"] = numpy.isnan(dictionary[i]["data"]).any()
        dictionary[i]["inf"] = numpy.isinf(dictionary[i]["data"]).any()
        dictionary[i]["events"] = eot.index[
            (
                (eot.loc[:, index] >= dictionary[i]["start"])
                & (eot.loc[:, index] <= dictionary[i]["end"])
                & (eot.loc[:, index] != dictionary[i]["datetime"])
            )
        ]

        for e in dictionary[i]["events"]:
            dictionary[e]["independent"] = False
            dictionary[e]["parent"] = i

        # unflagging events
        check[dictionary[i]["events"].union([i])] = False

        # appending indexer
        indexer.append(i)

    # peaks over threshold
    pot = eot.loc[indexer].set_index(index).assign(flag=None).sort_index()

    # computing timedelta
    pot.loc[:, "Δt"] = numpy.concatenate(
        [
            (
                (pot.index - lag)[1:].to_numpy()
                - (pot.index + lag)[:-1].to_numpy()
            )
            / pandas.Timedelta(hours=1),
            [numpy.nan],
        ]
    )

    # indexing events in soft margin
    events = pot.index[pot.loc[:, "Δt"] <= parameters[1] - 1]

    if len(events) > 0:
        for e_i, e_j in zip(events[:-1], events[1:]):
            pot.loc[
                pot.loc[:, column] == min(pot.loc[e_i:e_j].loc[:, column]),
                "flag",
            ] = False

    if numpy.any(pot.loc[:, "flag"].isna()):
        # maxima values
        maxima = pot.loc[pot.loc[:, "flag"].isna(), column]

    else:
        # nothing found
        maxima = None

    if output is None:
        return dataframe, maxima, threshold, parameters, dictionary
    elif output is True:
        return {
            "dataframe": dataframe,
            "maxima": maxima,
            "threshold": threshold,
            "parameters": parameters,
            "dictionary": dictionary,
        }
    elif isinstance(output, str):
        return locals()[output]
