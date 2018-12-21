import projet
import gspread
import clean
import time
from oauth2client.service_account import ServiceAccountCredentials

def next_available_row(worksheet):
    str_list = list(filter(None, worksheet.col_values(1)))  # fastest
    return len(str_list)+1

scope = ['https://spreadsheets.google.com/feeds',  'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
client = gspread.authorize(creds)

sh = client.open('results')
knnn = sh.worksheet("KNN n")


train_data, test_data = clean.importData()

traitees = clean.pretraitement(train_data["X"], val = 106, upper = 240, lower = 100, blackWhite = True)
traitessTest = clean.pretraitement(test_data["X"], val = 106, upper = 240, lower = 100, blackWhite = True)
nb = 15000
for n in range(3,30):

	line = clean.KNN(traitees[:nb], train_data['y'][:nb], traitessTest[:100], test_data['y'][:100], n)
	line.extend([nb, float(n)/float(nb)])
	knnn.insert_row(line, 2)

