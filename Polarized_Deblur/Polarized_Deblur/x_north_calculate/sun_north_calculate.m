clc; clear; close all;

%% ===================== 1. Basic parameter settings (adjust as needed) =====================
% 1.1 Observation point latitude and longitude (degrees) - Example is Cangzhou coordinates, replace with actual observation point
latitude_deg  = 38.57948;   % Latitude (positive for north latitude)
longitude_deg = 112.96385;  % Longitude (positive for east longitude)
L1 = deg2rad(latitude_deg); % Convert latitude to radians (required for astronomical calculations)

% 1.2 Folder path (stores .jpg files with timestamps, filename format: year_month_day_hour_minute_second.jpg)
folder_path = "D:\new";  % Replace with your image folder path

%% ===================== 2. Get and sort image files (by timestamp) =====================
% Get all .jpg files in the folder
image_files = dir(fullfile(folder_path, '*.jpg'));
num_images = length(image_files);
if num_images == 0
    error('No .jpg files in the specified folder, please check the path!');
end

% Sort by timestamp in filename (ensure calculation in chronological order)
timestamps = zeros(num_images, 1);
for i = 1:num_images
    fileName = image_files(i).name;
    [~, name, ~] = fileparts(fileName);
    parts = strsplit(name, '_'); % Filename format: year_month_day_hour_minute_second
    
    % Extract time and convert to date serial number (for sorting)
    if length(parts) >= 6
        try
            y = str2double(parts{1});
            m = str2double(parts{2});
            d = str2double(parts{3});
            h = str2double(parts{4});
            min_ = str2double(parts{5});
            s = str2double(parts{6});
            dt = datetime(y, m, d, h, min_, s);
            timestamps(i) = datenum(dt); % Date serial number (usable for sorting)
        catch
            timestamps(i) = Inf; % Files with incorrect format are placed at the end
        end
    else
        timestamps(i) = Inf; % Files with incorrect format are placed at the end
    end
end
% Sort files by timestamp in ascending order
[~, sortedIndices] = sort(timestamps);
image_files = image_files(sortedIndices);

%% ===================== 3. Initialize storage arrays =====================
allAzimuthAngles = zeros(num_images, 1); % Stores astronomical solar azimuth angle at each moment (degrees)
allhighAngles    = zeros(num_images, 1); % Stores astronomical solar altitude angle at each moment (degrees)
image_time_list  = cell(num_images, 1);  % Stores time corresponding to each file (for result verification)

%% ===================== 4. Core: Loop to calculate astronomical solar azimuth and altitude angles =====================
for img_idx = 1:num_images
    % 4.1 Parse time of current file (extract Beijing time from filename)
    img_name = image_files(img_idx).name;
    [~, name, ~] = fileparts(img_name);
    parts = strsplit(name, '_');
    
    % Verify filename format (year_month_day_hour_minute_second)
    if length(parts) < 6
        error(['Incorrect format for file ', num2str(img_idx), ': ', img_name, ', should follow "year_month_day_hour_minute_second.jpg"']);
    end
    % Extract time parameters (Beijing time)
    y = str2double(parts{1});  % Year
    m = str2double(parts{2});  % Month
    d = str2double(parts{3});  % Day
    h = str2double(parts{4});  % Hour (24-hour format)
    min_ = str2double(parts{5});% Minute
    s = str2double(parts{6});  % Second
    image_time_list{img_idx} = datetime(y, m, d, h, min_, s); % Record current time

    % 4.2 Convert Beijing time to UTC time (astronomical calculations require UTC, Beijing time is 8 hours ahead of UTC)
    beijingTime = datetime(y, m, d, h, min_, s, 'TimeZone', 'Asia/Shanghai');
    utcTime = beijingTime - hours(8); % UTC time = Beijing time - 8 hours
    % Extract year, month, day, hour, minute of UTC time (seconds have minimal impact on astronomical calculations and can be ignored)
    year1  = year(utcTime);
    month1 = month(utcTime);
    day1   = day(utcTime);
    hour1  = hour(utcTime);
    minute1= minute(utcTime);

    % 4.3 Calculate day of the year (dn) - considering leap years
    day_tab = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]; % Days per month in a common year
    if (mod(year1,4)==0 && mod(year1,100)~=0) || mod(year1,400)==0
        day_tab(3) = 29; % February has 29 days in a leap year
    end
    dn = sum(day_tab(1:month1)) + day1; % dn-th day of the year

    % 4.4 Calculate solar declination angle (ds) - Bourges algorithm (high precision, suitable for various latitudes)
    n0 = 78.801 + 0.2422*(year1 - 1969) - fix(0.25*(year1 - 1969));
    b = 2*pi*(dn - 1 - n0)/365.2422; % Parameter related to solar longitude
    ds = 0.3723 + 23.2567*sin(b) + 0.1149*sin(2*b) - 0.1712*sin(3*b) ...
         - 0.758*cos(b) + 0.3656*cos(2*b) + 0.0201*cos(3*b);
    ds = deg2rad(ds); % Convert declination angle to radians (required for subsequent calculations)

    % 4.5 Calculate time difference (Ts) - Difference between true solar time and mean solar time (unit: minutes)
    Ts = 0.0028 - 1.9587*sin(b) + 9.9059*sin(2*b) - 7.0924*cos(b) - 0.6882*cos(2*b);

    % 4.6 Calculate true solar time (ts) - Core time parameter for astronomical calculations (in radians)
    % Sd: Mean solar time (hours); st: True solar time (hours); ts: True solar time (radians, 12:00 as zero point)
    Sd = hour1 + 8.0 + (minute1 - (120.0 - longitude_deg)*4.0)/60.0;
    st = Sd + Ts/60;          % True solar time (hours)
    ts = (st - 12.0)*pi/12.0; % Convert true solar time to radians (range: -pi ~ pi)

    % 4.7 Calculate solar altitude angle (hs) - Angle from the horizon upwards (degrees)
    % Formula: sin(hs) = sin(declination angle)*sin(latitude) + cos(declination angle)*cos(latitude)*cos(true solar time)
    hs = asin(sin(ds)*sin(L1) + cos(ds)*cos(L1)*cos(ts));
    hs_deg = rad2deg(hs); % Convert radians to degrees, range: -90° (below horizon) ~ 90° (zenith)
    allhighAngles(img_idx) = hs_deg;

    % 4.8 Calculate solar azimuth angle (as1) - Angle clockwise from north (degrees)
    % Formula: cos(azimuth angle) = [sin(altitude angle)*sin(latitude) - sin(declination angle)] / [cos(altitude angle)*cos(latitude)]
    cos_as = (sin(hs)*sin(L1) - sin(ds)) / (cos(hs)*cos(L1));
    cos_as = min(max(cos_as, -1), 1); % Limit range (avoid errors in acos due to numerical inaccuracies)
    as_rad = acos(cos_as);            % Initial azimuth angle (radians)
    as_deg = rad2deg(as_rad);         % Convert to degrees

    % Azimuth angle correction (ensure compliance with "clockwise from north" geographic coordinate definition)
    as_deg = 180 - as_deg;                % First correction: adjust angle reference
    if ts > 0                             % Second correction: distinguish morning/afternoon (ts>0 is afternoon)
        as_deg = 360 - as_deg;
    end
    % Supplementary correction: angle compensation for times before 8:00 Beijing time (sun is eastward)
    if hour(beijingTime) < 8
        as_deg = 360 - as_deg;
    end
    % Ensure azimuth angle is within 0~360°
    as_deg = mod(as_deg, 360);
    allAzimuthAngles(img_idx) = as_deg;

    % Print current file's calculation results in real-time (for monitoring)
    fprintf('File %d: %s\n', img_idx, img_name);
    fprintf('  Observation time (Beijing time): %s\n', datestr(image_time_list{img_idx}, 'yyyy-mm-dd HH:MM:SS'));
    fprintf('  Astronomical solar altitude angle: %.2f° | Astronomical solar azimuth angle: %.2f°\n\n', hs_deg, as_deg);
end
