from pybaseball import playerid_lookup
from pybaseball import statcast_pitcher


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

