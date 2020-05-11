def get_zstr(snap):
    Snapshots = {'28': 'z000p000',
                 '25': 'z000p271',
                 '23': 'z000p503',
                 '21': 'z000p736',
                 '19': 'z001p004',
                 '17': 'z001p487',
                 '15': 'z002p012',
                 '13': 'z002p478',
                 '12': 'z003p017',
                 '11': 'z003p528'}
    return Snapshots[str(snap)]
