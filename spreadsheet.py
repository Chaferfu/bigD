import projet
import gspread
import clean
import time
from oauth2client.service_account import ServiceAccountCredentials
import dl_SVHM

def next_available_row(worksheet):
    str_list = list(filter(None, worksheet.col_values(1)))  # fastest
    return len(str_list)+1

scope = ['https://spreadsheets.google.com/feeds',  'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
client = gspread.authorize(creds)

sh = client.open('results')
lenet = sh.worksheet("lenet")
cnn = sh.worksheet("CNN")
mlp = sh.worksheet("MLP")

# lenet.insert_row(dl_SVHM.main("lenet"), 2)
# lenet.insert_row(dl_SVHM.main("lenet"), 2)
# lenet.insert_row(dl_SVHM.main("lenet"), 2)
# lenet.insert_row(dl_SVHM.main("lenet"), 2)
# lenet.insert_row(dl_SVHM.main("lenet"), 2)


lines = dl_SVHM.main("mlp", epoch_nbr = 5, batch_size = 10, learning_rate = 1e-3)
# lenet.insert_row(dl_SVHM.main("lenet", epoch_nbr = 10, batch_size = 7, learning_rate = 2.8e-3), 2)
for line in lines:
	mlp.insert_row(line, 2)