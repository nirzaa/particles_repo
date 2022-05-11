import scipy.io as matio

"""
Package 'scipy.io' includes the three most-often used function:
- savemat(FILENAME, output_dict):
  writes on file (overwrites if existed) with the values of 'output_dict'
  the keys (should be strings) will be used to identify the multiple values
  values will be transformed into array if it wasn't
- loadmat(FILENAME)
  return a 'input_dict' with the {key:value} combinations mentioned above
- whosmat(FILENAME)
  return a list of information as in tuples like (key, sizeOfValue_inTuple, typeOfValueElement)

For more information, see: [https://docs.scipy.org/doc/scipy/reference/io.html]
"""

import numpy

"""
Transformation functions between the ECAL's energy dataset of '(numpy.nd)array', 'dict', and 'list'
- for 'array', each cell is the energy deposit, and the indices are the coordinates as ECAL_array[(z,x,y)] = edep
- for 'dict', each entry is the energy deposit, and the key is the coordinates, also as ECAL_dict[(z,x,y)] = edep
  while for 'dict', the coordinates keys are NOT covering every cell but the cells with non-zero edep
- for 'list', each entry is a tuple containing the coordinates and the energy deposit as ECAL_list[n] = (z,x,y, edep)
  the first three elements of the entries could be duplicated, meaning they were created from different sources

NOTICE:
- the coordinates start from 'z' to 'x' and end with 'y', ZXY instead of XYZ
- the coordinates should be integers, but in some situations it will be converted to floats, like array(ECAL_list)
"""


def ecalDict2Array(edepdict):
    # creates an empty array and fills the numbers from the dict
    # array's more effecient for calculation
    edeparray = numpy.zeros([21, 110, 11])
    for key in edepdict: edeparray[key] = edepdict[key]
    return edeparray


def ecalArray2Dict(edeparray):
    # enumerates all elements in the array and fill a dict with non-zero ones
    # dict's more accurate for storage
    edepdict = {}
    for z in range(21):
        for x in range(110):
            for y in range(11):
                if edeparray[(z, x, y)] != 0: edepdict[(z, x, y)] = edeparray[(z, x, y)]
    return edepdict


def ecalList2Dict(edeplist):
    # transforms the tuple's first three elements to key and the third element to value
    # often be used in processing input. works for array-list as well
    edepdict = {}
    for edep in edeplist:
        # in the case of a duplicated/already-existed pad
        try:
            edepdict[tuple(map(int, edep[0:3]))] += edep[3]
        # in the case of a new pad
        except KeyError:
            edepdict[tuple(map(int, edep[0:3]))] = edep[3]
    return edepdict


def ecalDict2List(edepdict):
    # transforms the "key:entry" combinations into tuples
    # often be used in formatting output file
    edeplist = []
    for key in edepdict: edeplist.append(key + (edepdict[key],))
    return edeplist


def ecalmatio(filename, flagKeySensitive=True):
    # return a list of dicts of ecal-edep as {(z,x,y):energy}
    dictOfInput = {}  # if Key is sensitive
    listOfInput = []  # if Key is NOT sensitive
    inputdict = matio.loadmat(filename)
    # remove meta-data
    del inputdict['__globals__']
    del inputdict['__header__']
    del inputdict['__version__']

    if flagKeySensitive:  # return {eventid:{position:value}} double-dict, key turned into number instead of string
        i = 0
        for keyString in inputdict:
            # keyString_int = keyString
            # if type(keyString) == str:
            #     keyString_int = int(keyString)
            # dictOfInput[int(keyString)] = ecalList2Dict(inputdict[keyString].tolist())
            dictOfInput[keyString] = ecalList2Dict(inputdict[keyString].tolist())
            i += 1
        return dictOfInput
    else:  # read only the values and discard the keys, return [{position:value}] list of dicts
        for value in inputdict.values():
            listOfInput.append(ecalList2Dict(value.tolist()))
        return listOfInput


def xymatio(filename, flagKeySensitive=True):
    # return a list of dicts of ecal-edep as {(z,x,y):energy}
    dictOfInput = {}  # if Key is sensitive
    listOfInput = []  # if Key is NOT sensitive
    inputdict = matio.loadmat(filename)

    # remove meta-data
    del inputdict['__globals__']
    del inputdict['__header__']
    del inputdict['__version__']

    if flagKeySensitive:  # return {eventid:(x,y))} double-dict, key turned into number instead of string
        i = 0
        for keyString in inputdict:
            # keyString_int = keyString
            # if type(keyString) == str:
            #     keyString_int = int(keyString)
            # dictOfInput[int(keyString)] = tuple(inputdict[keyString][0:2].flatten())

            dictOfInput[keyString] = tuple(inputdict[keyString][0:2].flatten())
            i += 1
        return dictOfInput
    else:  # read only the values and discard the keys, return [(x,y)] list of dicts
        for keyString in inputdict.values():
            listOfInput.append(tuple(inputdict[keyString][0:2].flatten()))
        return listOfInput


def energymatio(filename) -> object:
    # return a dict of energies as {'id':(energy,...)}
    dictOfInput = {}  # if Key is sensitive, always True here
    inputdict = matio.loadmat(filename)
    # remove meta-data
    del inputdict['__globals__']
    del inputdict['__header__']
    del inputdict['__version__']
    i = 0
    for keyString in inputdict:
        dictOfInput[keyString] = tuple(inputdict[keyString][:, 0].flatten())
        i += 1
    return dictOfInput
