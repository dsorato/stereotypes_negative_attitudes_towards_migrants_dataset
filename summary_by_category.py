import os 
import sys 
import pandas as pd
def main(folder_path):
	path = os.chdir(folder_path)
	files = os.listdir(path)

	summary_by_category = pd.DataFrame(columns=['category', 'group',  'percentage_adverse', 'language', 'model'])

	groups = ['immigrants', 'refugees', 'foreigner', 'people', 'Null']
	for index, file in enumerate(files):
		if file.endswith("csv") and file!='summary_by_category.csv':
			language = file.split("_")[1]
			model = file.split("_")[-1]
			model = model.replace('.csv', '')

			print('###########', language, model, '###########')

			m_output = pd.read_csv(file, sep='\t')
			for g in groups:
				m_output_by_group = m_output[m_output['group'].str.contains(g)] 
				categorys = m_output_by_group.category.unique()
				for c in categorys:
					m_output_by_group_and_category = m_output_by_group[m_output_by_group['category'].str.contains(c)] 
					print('*********', g, c)
					print(m_output_by_group_and_category)

					percentage_adverse = len(m_output_by_group_and_category[m_output_by_group_and_category['score_sentence1']>m_output_by_group_and_category['score_sentence2']])/len(m_output_by_group_and_category)

					data = {'category':c, 'group':g,  'percentage_adverse':percentage_adverse*100, 'language':language, 'model':model}

					summary_by_category = summary_by_category.append(data, ignore_index = True)


					print(percentage_adverse)


	summary_by_category.to_csv('summary_by_category.csv', sep='\t', index=False)
	

if __name__ == "__main__":
	folder_path = str(sys.argv[1])
	main(folder_path)