import projet
import dl_SVHM
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def next_available_row(worksheet):
    str_list = list(filter(None, worksheet.col_values(1)))  # fastest
    return len(str_list)+1

scope = ['https://spreadsheets.google.com/feeds',  'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
client = gspread.authorize(creds)

sh = client.open('results')
cdm = sh.worksheet("CDM")
# cdm.insert_row(projet.CDM(False),next_available_row(cdm))

# print(trucs)

knn = sh.worksheet("KNN")

# # for i in range(1,16):
# # 	knn.insert_row(projet.KNN(True,i), next_available_row(knn))

# for i in range(13,16):
# 	knn.insert_row(projet.KNN(False,i), next_available_row(knn))


pre = sh.worksheet("preCDM")

# pre.insert_row(projet.testPretraitementCDM(25, 240, 100, True))

# for i in range(0, 256, 5):
# 	pre.insert_row(projet.testPretraitementCDM(i, 240, 100, True), 2)

# for i in range(0, 256, 5):
# 	pre.insert_row(projet.testPretraitementCDM(106, i, 100, True), 2)

# for i in range(0, 256, 5):
# 	pre.insert_row(projet.testPretraitementCDM(106, 240, i, True), 2)

# for i in range(45, 256, 5):
# 	pre.insert_row(projet.testPretraitementCDM(i, 240, 100, False), 2)

# for i in range(0, 256, 5):
# 	pre.insert_row(projet.testPretraitementCDM(106, i, 100, False), 2)

# for i in range(0, 256, 5):
# 	pre.insert_row(projet.testPretraitementCDM(106, 240, i, False), 2)

	
cnn = sh.worksheet("CNN")

# for i in range(1,20):
# 	cnn.insert_row(dl_SVHM.main(epoch_nbr = i), 2)

for i in range(10):
	cnn.insert_row(dl_SVHM.main(batch_size = 2**i))

for i in range(-5, 1):
	cnn.insert_row(dl_SVHM.main(learning_rate = 10**i))
