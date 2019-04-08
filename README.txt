############ files ###########

- ga.py
GA class: generatic algorithm (initiate hypothesis, operators: select,  mutation, crossover), convert chromosome to readable rules.

- run.py
parse inputfiles and  arguments 

- /data/tennis/*.txt
tennis data files for experiment "testTennis"

- /data/iris/*.txt
iris data files for experiments "testIris", "testIrisSelecton" and "testIrisReplacement"

############# Usage ###########
python3 run.py -e [experiment: testTennis, testIris, testIrisSelecton, testIrisReplacement]

'''
e.g.
python3 run.py -e testTennis
'''

######## Dependencies ########
pandas
numpy

