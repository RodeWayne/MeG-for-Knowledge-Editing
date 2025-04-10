# add new id to cf dataset
python add_id_for_cf.py

# get new zsre dataset
python get_new_zsre.py

# get new rephrase for zsre dataset
# it may take a long time
# and result maybe different because of the randomness of LLMs 
# you can skip it if you have downloaded our new split data
python zsre_get_new_rephrase.py

