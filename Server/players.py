from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup

def return_player_data(player_name, start_date, end_date):
    """
    Returns the player_data id of a player_data based on the player_data name.
    """
    name_splitted = player_name.split()
    first_name = name_splitted[0]
    last_name = name_splitted[1]
    try:
        player_id = playerid_lookup(last_name, first_name)['key_mlbam'][0]
        player_data = statcast_pitcher(start_date, end_date, player_id=player_id)[['type', 'plate_x', 'plate_z']]
        print(player_data)
        if player_data.empty:
            raise ValueError('No data for this player')
    except:
        player_id = playerid_lookup('Sale', 'Chris')['key_mlbam'][0]
        player_data = statcast_pitcher('2008-04-01', '2017-07-15', player_id=player_id)[['type', 'plate_x', 'plate_z']]
    return player_data


#return_player_data('Chris sale')

# return player_id
# player_data = playerid_lookup('Judge', 'Aaron')
# print(player_data.columns)
# playerid = player_data['key_mlbam'][0]
# print(playerid)
#
# aaron_judge = statcast_pitcher('2008-04-01', '2017-07-15', player_id=playerid)
# print(aaron_judge.head())
# print(aaron_judge.columns)