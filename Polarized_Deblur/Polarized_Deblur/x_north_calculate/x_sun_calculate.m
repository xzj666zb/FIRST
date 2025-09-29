clc; clear; close all;

%% ===================== 1. Basic parameter settings =====================
folder_path = "D:\new"; % Polarization image folder path
output_folder = 'D:\524\output'; % Result output folder path
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% ===================== 2. Acquire and sort polarization images by time =====================
image_files = dir(fullfile(folder_path, '*.jpg'));
num_images = length(image_files);
if num_images == 0
    error('No .jpg polarization images found in the specified folder. Please check the path or file format filter!');
end

% Sort by filename timestamp
timestamps = zeros(num_images, 1);
for i = 1:num_images
    fileName = image_files(i).name;
    [~, name, ~] = fileparts(fileName);
    parts = strsplit(name, '_');
    
    if length(parts) >= 6
        try
            y = str2double(parts{1});
            m = str2double(parts{2});
            d = str2double(parts{3});
            h = str2double(parts{4});
            min_ = str2double(parts{5});
            s = str2double(parts{6});
            dt = datetime(y, m, d, h, min_, s);
            timestamps(i) = datenum(dt);
        catch
            timestamps(i) = Inf;
        end
    else
        timestamps(i) = Inf;
    end
end

[~, sortedIndices] = sort(timestamps);
image_files = image_files(sortedIndices);

%% ===================== 3. Initialize storage arrays =====================
all_sun_azimuths = zeros(num_images, 1);

%% ===================== 4. Core: loop over polarization images to compute solar azimuth =====================
for img_idx = 1:num_images
    % 4.1 Get current image path and name
    image_path = fullfile(folder_path, image_files(img_idx).name);
    img_name = image_files(img_idx).name;
    img_name_prefix = img_name(1:end-4);
    
    % 4.2 Read polarization image and convert to grayscale
    img = imread(image_path);
    img = double(im2gray(img));

    % 4.3 Extract four polarization channels
    img0 = img(2:2:end, 2:2:end);   % 0° polarization component
    img45 = img(1:2:end, 2:2:end);  % 45° polarization component
    img90 = img(1:2:end, 1:2:end);  % 90° polarization component
    img135 = img(2:2:end, 1:2:end); % 135° polarization component

    % 4.4 Compute Stokes parameters
    I = (img0 + img45 + img90 + img135) / 2;
    Q = img0 - img90;
    U = img45 - img135;

    % 4.5 Compute angle of polarization (AOP) and degree of polarization (DOP)
    AOP = 0.5 * atan2(U, Q);  % AOP in radians
    DOP = sqrt(Q.^2 + U.^2) ./ max(I, 1e-10); % DOP

    % 4.6 Transform AOP to local meridian coordinate system
    [height_AOP, width_AOP] = size(AOP);
    center_x = floor(width_AOP / 2);
    center_y = floor(height_AOP / 2);
    [x_coords, y_coords] = meshgrid(1:width_AOP, 1:height_AOP);
    phi = atan2(center_y - y_coords, x_coords - center_x); % Pixel azimuth relative to center
    AOP_local = AOP - phi; % Transform to local meridian frame
    
    % Adjust AOP_local range to [-π/2, π/2]
    AOP_local(AOP_local < -pi/2) = AOP_local(AOP_local < -pi/2) + pi;
    AOP_local(AOP_local > pi/2) = AOP_local(AOP_local > pi/2) - pi;

    % 4.7 Directly use atmospheric polarization pattern
    % Use AOP_local as atmospheric AOP
    P_air = AOP_local;

    % 4.8 Determine sun position based on CPP (consistency of polarization angle)
    % Create valid pixel mask (use entire image)
    valid_mask = true(size(P_air));
    valid_indices = find(valid_mask);
    num_pixels = length(valid_indices);
    E_matrix = zeros(3, num_pixels);

    % Compute incident angle based on pixel position (assume camera focal length)
    [height_img, width_img] = size(img);
    center_x2 = floor(width_img / 2);
    center_y2 = floor(height_img / 2);
    [x_coords2, y_coords2] = meshgrid(1:width_img, 1:height_img);
    
    % Compute incident angle
    focal_length = 8 / 0.00345; % Assumed focal length
    r = sqrt((y_coords2 - center_y2).^2 + (x_coords2 - center_x2).^2) / focal_length;
    r = atan(r); % Incident angle
    
    % Downsample incident angle to match AOP_local size
    r_small = r(1:2:end, 1:2:end);

    % Loop over all valid pixels to build direction vector matrix
    for idx = 1:num_pixels
        [y_pix, x_pix] = ind2sub(size(valid_mask), valid_indices(idx));
        chi_a = P_air(y_pix, x_pix); % Atmospheric AOP
        dop = DOP(y_pix, x_pix);     % DOP
        i_pixel = r_small(y_pix, x_pix); % Incident angle
        phi_pixel = phi(y_pix, x_pix);   % Azimuth

        % Build polarization direction vector in local frame
        e_l = [cos(chi_a), sin(chi_a), 0]';
        
        % Coordinate transformation matrix: local to global
        C_ia = [cos(i_pixel)*cos(phi_pixel), -sin(phi_pixel), sin(i_pixel)*cos(phi_pixel);
                cos(i_pixel)*sin(phi_pixel),  cos(phi_pixel), sin(i_pixel)*sin(phi_pixel);
               -sin(i_pixel),                0,              cos(i_pixel)];
        
        e_a = C_ia * e_l;             % Global polarization direction vector
        e_a_star = dop * e_a;         % Weighted direction vector

        E_matrix(:, idx) = e_a_star;
    end

    % 4.9 Eigenvalue decomposition to obtain sun direction
    EEt = E_matrix * E_matrix';
    [eigen_vectors, eigen_values] = eig(EEt);
    lambda = diag(eigen_values);
    [~, min_idx] = min(lambda);
    s_prime = eigen_vectors(:, min_idx);

    % Adjust z-component sign of sun direction vector (ensure pointing to sky)
    z_axis = [0; 0; 1];
    s = sign(dot(s_prime, z_axis)) * s_prime;

    % 4.10 Compute solar azimuth
    phi_s = atan2(s(2), s(1));
    phi_s_deg = mod(rad2deg(phi_s) + 360, 360);
    all_sun_azimuths(img_idx) = phi_s_deg;

    % 4.11 Save result
    result_file = fullfile(output_folder, [img_name_prefix '_polar_sun_azimuth.txt']);
    fid = fopen(result_file, 'w');
    fprintf(fid, 'Polarization-based solar azimuth: %.2f°\n', phi_s_deg);
    fclose(fid);

    % 4.12 Print progress in real time
    fprintf('Processed image %d/%d: %s\n', img_idx, num_images, img_name);
    fprintf('  Polarization-based solar azimuth: %.2f°\n\n', phi_s_deg);
end

%% ===================== 5. Final result output =====================
fprintf('===================== Summary of polarization-based solar azimuths =====================\n');
for idx = 1:num_images
    fprintf('Image %d (%s): polarization-based solar azimuth = %.2f°\n', ...
        idx, image_files(idx).name, all_sun_azimuths(idx));
end

disp(' ');
disp('All polarization images processed! Solar azimuths stored in all_sun_azimuths array, individual result files saved to output folder.');
