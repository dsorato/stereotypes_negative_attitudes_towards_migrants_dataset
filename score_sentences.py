from transformers import AutoTokenizer, RobertaForMaskedLM
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM,AutoModelWithLMHead
import torch
import pandas as pd
import sys
from transformers import XGLMTokenizer, XGLMForCausalLM
import spacy


#python -m spacy download pt_core_news_sm
#python -m spacy download ca_core_news_sm
#python -m spacy download es_core_news_sm
stereo_score = 0

nlp_pt = spacy.load('pt_core_news_sm')
nlp_es = spacy.load('es_core_news_sm')
nlp_ca = spacy.load('ca_core_news_sm')

def get_gender_and_plural_form(word, language):
    if language.lower() == 'pt':
        doc = nlp_pt(word)
    elif language.lower() == 'es':
        doc = nlp_es(word)
    else:
        doc = nlp_ca(word)

    is_plural = any('Number=Plur' in token.morph for token in doc)

    gender = None
    if any('Gender=Fem' in token.morph for token in doc):
        gender = 'fem'
    else:
         gender = 'masc'

    return gender, is_plural

def get_groups(row_groups):
	ret = []

	groups = row_groups.split(',')

	for g in groups:
		ret.append(g.lstrip(' '))

	return ret

def replace_ap_in_sentence_according_group(row, sentence, group, gender):
	if gender == 'fem' and row['art_pron_fem'] != 'Null':
		sentence = sentence.replace('[AP]', row['art_pron_fem'])
	else:
		sentence = sentence.replace('[AP]', row['art_pron_masc'])

	return sentence


def replace_prep_in_sentence_according(row, sentence, preposition):
	if preposition == '_contracted' and row['preposition_contracted'] != 'Null':
		sentence = sentence.replace('[PREP]', row['preposition_contracted'])
	else:
		sentence = sentence.replace('[PREP]', row['preposition_not_contracted'])

	return sentence
	


def replace_concept_in_sentence(row, sentence, gender):
	if gender == 'fem' and row['concept1_fem'] != 'Null':
		sentence1 = sentence.replace('[CONCEPT]', row['concept1_fem'])
		sentence2 = sentence.replace('[CONCEPT]', row['concept2_fem'])

		concept1 = row['concept1_fem']
		concept2 = row['concept2_fem']
	else:
		sentence1 = sentence.replace('[CONCEPT]', row['concept1_masc'])
		sentence2 = sentence.replace('[CONCEPT]', row['concept2_masc'])

		concept1 = row['concept1_masc']
		concept2 = row['concept2_masc']

	return sentence1, concept1, sentence2, concept2



def sentence_score_with_group(row, sentence, sentence_template, gender, group, tokenizer, model, results, stereo_score_dict, stereo_groups_dict):
	log_softmax = torch.nn.LogSoftmax(dim=1)
	print(sentence)

	if group.startswith('refugi'):
		simplified_group = 'refugees'
	elif group.startswith('inmigrante') or group.startswith('immigrant') or group.startswith('imigrante'):
		simplified_group = 'immigrants'
	elif group.startswith('extranjero') or group.startswith('estrang'):
		simplified_group = 'foreigner'
	else:
		simplified_group = 'people'

	if simplified_group in stereo_groups_dict:
		stereo_groups_dict[simplified_group] +=1
	else:
		stereo_groups_dict[simplified_group] = 1

	sentence1, concept1, sentence2, concept2 = replace_concept_in_sentence(row, sentence, gender)

	sentence1_token_ids = tokenizer.encode(sentence1, return_tensors='pt')
	sentence2_token_ids = tokenizer.encode(sentence2, return_tensors='pt')
	
	with torch.no_grad():
		score_sentence1 = calculate_aul_for_bert(model, sentence1_token_ids, log_softmax)
		score_sentence2 = calculate_aul_for_bert(model, sentence2_token_ids,log_softmax)

	if score_sentence1 > score_sentence2:
		if simplified_group in stereo_score_dict.keys():
			stereo_score_dict[simplified_group] +=1
		else:
			stereo_score_dict[simplified_group] = 1
		most_probable_concept = concept1
	else:
		most_probable_concept = concept2


	data = {'templateId': row['templateId'], 'category':row['category'], 'subcategory':row['subcategory'],
		    'group': simplified_group, 'sentence_template':sentence_template, 
			'most_probable_concept':most_probable_concept, 'concept1':concept1, 'concept2':concept2, 
			'score_sentence1': score_sentence1, 'score_sentence2':score_sentence2}

	results = results.append(data, ignore_index = True)

	return results, stereo_score_dict, stereo_groups_dict



def sentence_score_without_group(row, sentence, sentence_template, gender, tokenizer, model, results, stereo_score):
	print(sentence)
	log_softmax = torch.nn.LogSoftmax(dim=1)

	sentence1, concept1, sentence2, concept2 = replace_concept_in_sentence(row, sentence, gender)

	sentence1_token_ids = tokenizer.encode(sentence1, return_tensors='pt')
	sentence2_token_ids = tokenizer.encode(sentence2, return_tensors='pt')
	
	with torch.no_grad():
		score_sentence1 = calculate_aul_for_bert(model, sentence1_token_ids, log_softmax)
		score_sentence2 = calculate_aul_for_bert(model, sentence2_token_ids,log_softmax)

	if score_sentence1 > score_sentence2:
		stereo_score += 1
		most_probable_concept = concept1
	else:
		most_probable_concept = concept2


	data = {'templateId': row['templateId'], 'category':row['category'], 'subcategory':row['subcategory'],
		    'group': 'Null', 'sentence_template':sentence_template, 
			'most_probable_concept':most_probable_concept, 'concept1':concept1, 'concept2':concept2, 
			'score_sentence1': score_sentence1, 'score_sentence2':score_sentence2}

	results = results.append(data, ignore_index = True)

	return results, stereo_score	




def calculate_aul_for_bert(model, token_ids, log_softmax):
	'''
	Function Implementation from https://github.com/kanekomasahiro/evaluate_bias_in_mlm/
	Given token ids of a sequence, return the averaged log probability of
	unmasked sequence (AUL).
	'''
	output = model(token_ids)
	logits = output.logits.squeeze(0)
	log_probs = log_softmax(logits)
	token_ids = token_ids.view(-1, 1).detach()
	token_log_probs = log_probs.gather(1, token_ids)[1:-1]
	log_prob = torch.mean(token_log_probs)
	log_prob = log_prob.item()

	return log_prob

def main(language):
	#Catalan=ca
	#Portuguese=pt
	#Spanish=es
	
	language = language.lower()
	dataset = pd.read_csv("./datasets/dataset_"+language+".csv", sep='\t')


	#model_name='PlanTL-GOB-ES/roberta-base-ca'
	#model_name='PORTULAN/albertina-ptpt'
	# model_name='PlanTL-GOB-ES/roberta-large-bne'

	#model_name='Twitter/twhin-bert-base'
	#model_name='xlm-roberta-base'
	#model_name='distilbert-base-multilingual-cased'

	# model_name='projecte-aina/FLOR-1.3B'
	# model_name='ai-forever/mGPT'
	#model_name='bigscience/bloom-1b1'


	if model_name.startswith('bigscience/bloom') or model_name.startswith('projecte-aina/FLOR-1.3B') or model_name.startswith('ai-forever/mGPT'):
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		model = AutoModelForCausalLM.from_pretrained(model_name)
	else:
		model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
		tokenizer = AutoTokenizer.from_pretrained(model_name)




	stereo_score_dict = dict()
	stereo_groups_dict = dict()

	stereo_score = 0


	# dict_groups_gendered = {'inmigrante': 'masc', 'imigrante': 'masc', 'refugiado': 'masc', 'extranjero': 'masc', 'estrangeiro': 'masc', 'persona': 'fem',  'pessoa': 'fem',
	# 'immigrant': 'masc', 'refugiat': 'masc', 'estranger': 'masc', 'persones':'fem', 'expatriado': 'masc', 'expatriat': 'masc'}

	results = pd.DataFrame(columns=['templateId', 'category', 'subcategory', 'group',  'sentence_template', 'most_probable_concept', 'concept1', 'concept2', 'score_sentence1', 'score_sentence2'])

	for i, row in dataset.iterrows():
		if  '[GROUP]' in row['sentence']:
			groups = get_groups(row['group'])
			for g in groups:
				sentence_template = row['sentence']
				gender, is_plural = get_gender_and_plural_form(g, language)
				#gender = [v for k, v in dict_groups_gendered.items() if g.startswith(k)][0]
				sentence = sentence_template.replace('[GROUP]', g)
				if language=='ca':
					if g[0] in ['a', 'e', 'i', 'o', 'u', 'h']:
						preposition='_contracted'
					else:
						preposition='_not_contracted'
					# preposition = [v for k, v in dict_groups_preposition_contraction.items() if g.startswith(k)][0]
					sentence = replace_prep_in_sentence_according(row, sentence, preposition)
				if '[AP]' in sentence:
					sentence = replace_ap_in_sentence_according_group(row, sentence, g, gender)

				
				results, stereo_score_dict, stereo_groups_dict = sentence_score_with_group(row, sentence, sentence_template, gender, g, tokenizer, model, results, stereo_score_dict, stereo_groups_dict)
		else:
			sentence_template = row['sentence']
			gender = 'masc'
			group = 'Null'

			results, stereo_score = sentence_score_without_group(row, sentence, sentence_template, gender, tokenizer, model, results, stereo_score)


	if '/' in model_name:
		model_name=model_name.split('/')[-1]


	templates_without_group = (dataset.group.str.contains('Null')).sum()

	for k, v in list(stereo_score_dict.items()):

		percentage_group = round((v / stereo_groups_dict[k]) * 100, 2)
		print('Bias score for "'+k+'" group: ', percentage_group)
		#print(v, stereo_groups_dict[k])

	percentage_group_no_group = round((stereo_score / templates_without_group) * 100, 2)
	#print(stereo_score, templates_without_group)
	print('Bias score for no groups: ', percentage_group_no_group)

	results.to_csv('results_'+language+'_'+model_name+'.tsv', sep='\t', index=False)



if __name__ == "__main__":
	language = str(sys.argv[1])
	main(language)