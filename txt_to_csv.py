"""
Take .txt files from `nz-debates/` and convert to a csv file split by message with speaker name and date
"""
# %%
from os import listdir
from os.path import isfile, join

import pandas as pd
# %%
fld = '../nz-debates/'
# %%
txt_fns = [f for f in listdir(fld) if isfile(join(fld, f))]
# %%
### parse into one row per message
def get_speaker(ln):
    if ':' in ln[:75]:
        return ln.split(':')[0]
    else:
        return None


def get_message(ln):
    if ':' in ln[:75]:
        return ':'.join(ln.split(':')[1:]).strip(' ')
    else:
        return ln.strip(' ')


def parse_txt_file(fn):
    df_day = pd.DataFrame(columns=['date', 'block', 'speaker', 'message'])
    date = f'{fn[:4]}-{fn[4:6]}-{fn[6:8]}'
    with open(fld + fn) as file:
        lines = file.readlines()
        text = '/n'.join([l.rstrip() for l in lines])
    # split into blocks
    while '/n/n/n' not in text:
        text = text.replace('/n/n/n', '/n/n')
    blocks = text.split('/n/n')
    blocks = [b for b in blocks if b != '']
    # parse the blocks
    for block_n, block in enumerate(blocks):
        is_intro = True
        for line in block.split('/n'):
            speaker = get_speaker(line)
            # intro is defined as block, up until a speaker is found
            if speaker != None:
                is_intro = False
            # disregard the intro (atleast for now)
            if is_intro:
                continue
            # if is not intro and there is no speaker, assume it is the same speaker
            elif speaker == None:
                speaker = previous_speaker
            else:
                previous_speaker = speaker
            # Get the message and add a row
            message = get_message(line)
            row = {'date': [date], 'block': [block_n], 'speaker': [speaker], 'message': [message]}
            df_row = pd.DataFrame(row)
            df_day = pd.concat([df_day, df_row])
    return df_day


fn = '20201201.txt'
df_day = parse_txt_file(fn)
# %%
# df_day.to_csv('TEMP_day.csv', index=False)
# %%
# df_day['speaker'].value_counts()[-15:]
# %%
df = pd.DataFrame(columns=['date', 'block', 'speaker', 'message'])
for fn in txt_fns:
    print(f'Parsing {fn}')
    df_day = parse_txt_file(fn)
    df = pd.concat([df, df_day])
# %%
df.shape
# %%
df['speaker'].value_counts()[:5]
# %%
df.to_csv('debates_by_message.csv', index=False)
# %%
