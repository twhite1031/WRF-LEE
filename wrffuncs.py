import glob
import os
from datetime import datetime, timedelta
# Adjust datetime to match filenames
def round_to_nearest_5_minutes(dt):
    # Extract the minute value
    minute = dt.minute

    # Calculate the nearest 5-minute mark
    nearest_5 = round(minute / 5) * 5

    # Handle the case where rounding up to 60 minutes
    if nearest_5 == 60:
        dt = dt + timedelta(hours=1)
        nearest_5 = 0

    # Replace the minute value with the nearest 5-minute mark
    rounded_dt = dt.replace(minute=nearest_5, second=0, microsecond=0)

    return rounded_dt

def parse_filename_datetime_obs(filepath):
    
    # Get the filename 
        filename = filepath.split('/')[-1]
    # Extract the date part (8 characters starting from the 5th character)
        date_str = filename[4:12]
    # Extract the time part (6 characters starting from the 13th character)
        time_str = filename[13:19]
    # Combine the date and time strings
        datetime_str = date_str + time_str
    # Convert the combined string to a datetime object
    #datetime_obj = datetime.datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
    #formatted_datetime_obs = parse_filename_datetime_obs(radar_data)datetime_obj.strftime('%B %d, %Y %H:%M:%S') 
        return datetime.strptime(datetime_str, '%Y%m%d%H%M%S')

def find_closest_radar_file(target_datetime, directory, radar_prefix=None):
    """Finds the file in the directory with the datetime closest to the target datetime."""
    closest_file = None
    closest_diff = None
    
    # Iterate over all files in the directory
    if radar_prefix:
        search_pattern = os.path.join(directory, f'{radar_prefix}*.ar2v')
    else:
        search_pattern = os.path.join(directory, '*.ar2v')

    for filepath in glob.glob(search_pattern):
        # Extract the filename
        filename = os.path.basename(filepath)
        try:
            # Parse the datetime from the filename
            file_datetime = parse_filename_datetime_obs(filename)
            # Calculate the difference between the file's datetime and the target datetime
            diff = abs((file_datetime - target_datetime).total_seconds())
            # Update the closest file if this file is closer
            if closest_diff is None or diff < closest_diff:
                closest_file = filepath
                closest_diff = diff
        except ValueError:
            # If the filename does not match the expected format, skip it
            continue
    
    return closest_file

def parse_filename_datetime_wrf(filepath, timeidx):
    
    # Define the format of the datetime string in your filename
        datetime_format = "wrfout_d02_%Y-%m-%d_%H:%M:%S"

    # Parse the datetime string into a datetime object
        time_object = datetime.strptime(os.path.basename(filepath), datetime_format)

    # Add timeidx value
        add_time = 5 * timeidx
        time_object_adjusted = time_object + timedelta(minutes=add_time)

        return time_object_adjusted

