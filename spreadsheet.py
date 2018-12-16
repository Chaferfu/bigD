import projet
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
cdm.insert_row(projet.CDM(True),next_available_row(cdm))

print(trucs)