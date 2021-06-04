//-----------------------------------------------------------------------
// PREPARE TILES FOR MIST STITCHING
// Created 09-03-2021 by Lukas van den Heuvel.
//
// The CX7 high-content microscope arranges its images in a spiral,
// and stores each color channel in a separate image.
//
// This macro converts the filenames from a spiral grid into a column grid,
// so that Fiji can use it for column-grid stitching.
// Furthermore, it performs operations on seperate color channels 
// (e.g. background subtraction, contrast stretching, histogram equalization),
// and saves tiles in the format tile_{ppp}_ch{t} (where ppp is the position
// in the column grid, and t is the channel nr.)
//
// Additionally, there is the possibility to set a threshold on centrain channels.
//
// The output is a folder inside the root folder of the experiment
// with the name of the well.
// Inside this well_folder a folder "tiles" is created. It contains one image per tile channel.
// The names of the tiles are ordered in a column grid.
// If you also asked the computer to threshold tiles, the thresholds can be found in 'tiles/thresholds'.
//-----------------------------------------------------------------------

//---------------------------START FUNCTIONS-----------------------------

function initialize_position(width, height, clockwise){

	//-----------------------------------------------------
	// Finds the center of a spiral grid, i.e. the 
	// (x,y) location in the spiral grid with value 0.

	// Inputs
	// width & height: the dimensions of the spiral grid.

	// Output
	// init_position: array of length 2. 
	// Index 0 is the x position, index 1 is the y position.
	//-----------------------------------------------------
	add_y = 0;
	if(clockwise == false){
		add_y = 1;
	}
	
	init_position = newArray(2);
	if (width%2 == 0){
		x = width / 2 - 1;
	}else{
		x = floor(width / 2);
	}
	if (height%2 == 0){
		y = height / 2 - 1 + add_y;
	}else{
		y = floor(height / 2);
	}
	init_position[0] = x;
	init_position[1] = y;

	return init_position;

}

function compare_arrays(arr1, arr2){

	//----------------------------------------------------------
	// Function compares array arr1 and array arr2 and
	// returns true if they are equal, and false if they are not.
	//----------------------------------------------------------
	
	same = true;
	l = arr1.length;
	for (i = 0; i < l; i++){
		if (arr1[i] != arr2[i]){
			same = false;
		}
	}
	return same;
}

function turn_right(current_direction, clockwise){

	//-----------------------------------------------------
	// This function is called by make_spiral_grid().
	
	// It takes as input the current direction (north, south, west and east),
	// and outputs the direction after turning right:
	// if clockwise:
	// north->east, south->west, east->south, west->north.
	// else (counterclockwise):
	// north->west, south->east, east->north, west->south.
	
	// All directions are arrays of length 2:
	// index 0 is dx, index 1 is dy.
	//-----------------------------------------------------
	
	NORTH = newArray(0,-1);
	S = newArray(0,1);
	W = newArray(-1,0);
	E = newArray(1,0);

	if (clockwise){
		if (compare_arrays(current_direction, NORTH)){
			new_direction = E;
		}else if (compare_arrays(current_direction, S)){
			new_direction = W;
		}else if (compare_arrays(current_direction, E)){
			new_direction = S;
		}else if (compare_arrays(current_direction, W)){
			new_direction = NORTH;
		}
		
	}else{
		if (compare_arrays(current_direction, NORTH)){
			new_direction = W;
		}else if (compare_arrays(current_direction, S)){
			new_direction = E;
		}else if (compare_arrays(current_direction, E)){
			new_direction = NORTH;
		}else if (compare_arrays(current_direction, W)){
			new_direction = S;
		}
	}

	return new_direction;
	
}

function make_spiral_grid(width,height,clockwise){
	//-----------------------------------------------------
	// This function makes a spiral grid array.
	// Note that 2D arrays are not supported by Fiji macro language.
	// The output will be a 1D array with length width*height.
	
	// If you index it with matrix[x + y * width], 
	// the matrix is effectively 2D.
	//-----------------------------------------------------
	
	// Check if dimenstions are at least 1:
	if (width < 1 || height < 1){
		return;
	}

	// Directions to walk (dx, dy) are arrays:
	NORTH = newArray(0,-1);
	S = newArray(0,1);
	W = newArray(-1,0);
	E = newArray(1,0);

	// Initial position:
	init_position = initialize_position(width, height, clockwise);
	x = init_position[0];
	y = init_position[1];

	// We want to start walking to the west. 
	// This means our initial direction is north (clockwise) or south (counterclockwise):
	// we then turn right immediately, and end up going west.
	if(clockwise){
		direction = NORTH;
	}else{
		direction = S;
	}
	dx = direction[0];
	dy = direction[1];

	// Initialize matrix:
	matrix = newArray(w*w);
	Array.fill(matrix, NaN);
	count = 0;

	while (true){
		// Fill matrix at current position with value count:
		matrix[x + y*width] = count;
		
		count = count + 1;
		
		// Try to turn right:
		new_direction = turn_right(direction, clockwise);
		new_dx = new_direction[0];
		new_dy = new_direction[1];
		new_x = x + new_dx;
		new_y = y + new_dy;

		// Turn right if you are not yet at the boundaries,
		// and if the position at your right-hand side is not yet visited:
		if ((0<=new_x) & (new_x<width) & (0<=new_y) & (new_y<height) & (isNaN(matrix[new_x + new_y*width]))){
			x = new_x;
			y = new_y;
			dx = new_dx;
			dy = new_dy;
			direction = new_direction;
		}
		// If not, go straight:
		else{
			x = x + dx;
			y = y + dy;
			// If you are at the boundary, stop the process and return the matrix:
			if (!(((0 <= x) & (x < width)) & (0 <= y) & (y < height))){
				return matrix;
			}
		}
	}
}

function make_column_grid(width, height){

	//-----------------------------------------------------
	// This function makes a column grid array.
	// Note that 2D arrays are not supported by Fiji macro language.
	// The output will be a 1D array with length width*height.
	
	// If you index it with matrix[x + y * width], 
	// the matrix is effectively 2D.
	//-----------------------------------------------------
	
	matrix = newArray(width*height);
	Array.fill(matrix, NaN);
	count = 0;
	x = 0;
	y = 0;
	for (i = 0; i < width*height; i++) {
		matrix[x + y*width] = count;
		count = count + 1;
		y = y + 1;
		if (y == height){
			y = 0;
			x = x + 1;
		}
	}
	return matrix;
}

function find_ch_operation_boolean(nr_channels, ch_operation_string){
	
	//----------------------------------------------------------------------------------
	// This function is used to make a user choose on which color 
	// channels they want to do an operation.

	// Examples of operations are: background subtraction, 
	// contrast enhancement, histogram equalization.

	// The user enters the channels into the dialog window,
	// seperated by commas.
	// This results in a string, example: ch_enhance_string = "1,2".
	// This function converts the string in a boolean with length
	// nr_channels (=number of channels).

	// So if nr_channels=3, the function outputs ch_enhance_boolean = [false, true, true].
	//----------------------------------------------------------------------------------

	// Split the string on commas:
	ch_operation_split = split(ch_operation_string, ",");
	// Initialize a boolean array with length nr_channels:
	ch_operation_boolean = newArray(nr_channels);

	// Loop over all channels
	for (nr = 0; nr < nr_channels; nr++){

		do_operation = false;
		
		// if this channel appears in ch_operation_string, make do_operation = true.
		for (i = 0; i < ch_operation_split.length; i++) {
			ch_nr = ch_operation_split[i];
			if (d2s(nr,0) == ch_nr){
				do_operation = true;
			}
		}
		ch_operation_boolean[nr] = do_operation;
	}
	return ch_operation_boolean;
}

function get_next_checkbox_group(nr_channels){
	// This function outputs a boolean array with
	// checkbox choices of a checkbox group.
	bool = newArray(nr_channels);
	for (i=0; i<nr_channels; i++){
		bool[i] = Dialog.getCheckbox();
	}
	return bool;
}

function find_name_of_first_image_in_list(file_list){
	
	//---------------------------------------------------------------------
	// Some files in a raw directory might not be images.
	// This function finds the name of the first HCA image in the directory.
	// This name serves as a template to find other files.
	//---------------------------------------------------------------------
	
	for (i = 0; i < file_list.length; i++) {
		fileName = file_list[i];
		splitFileName = split(fileName, "_");
		if (splitFileName[0] == "MFGTMP") { // The file is an image made by the HCA
			img0 = fileName;
			break
		}
	}
	return img0;
}

function number_to_string(nr, min_length){
	//------------------------------------------------------
	// This function converts an integer nr into a string.
	// Examples: 
	// if min_length is 2:
	// 0 --> "00", 3 --> "03", 11 --> "11", 102 --> "102".
	// If min_length is 3:
	// 0 --> "000", 3 --> "003", 11 --> "011", 102 --> "102".
	//------------------------------------------------------

	nr_string = d2s(nr,0);
	number_of_zeros_to_add = min_length - lengthOf(nr_string);
	if (number_of_zeros_to_add>0){
		zeros = "";
		for (i = 0; i < number_of_zeros_to_add; i++) {
			zeros = zeros + "0";
		}
		nr_string = zeros + nr_string;
	}
	return nr_string;
}

function get_mfgtmp_file_name(file0, well, nr, ch){
	
	//------------------------------------------------------
	// This function returns the filename of a raw tile
	// generated by the CX7 high content microscope.

	// Inputs
	// ------
	// file0: string
	// Name of the first file in the image folder. This serves as a template.

	// well: string
	// Name of the well, e.g. "B02".

	// nr: int
	// Number of the tile in the spiral grid.

	// ch: int
	// Channel number.

	// Output
	// ------
	// file_name: string
	// Name of the file specified by well, nr and ch.
	// Example: file_name = "MFGTMP_201029210001_B03f00d0.TIF"
	//------------------------------------------------------
	
	parts = split(file0, "_");
	nr_string = number_to_string(nr,2);
	file_name = parts[0] + "_" + parts[1] + "_" + well + "f" + nr_string + "d" + d2s(ch,0) + ".TIF";
	return file_name;
}

function get_file_name_from_format(filename_format, well, img_nr, ch){
	//---------------------------------------------
	// This function returns the filename as it is specified by a filename format.
	// A filename format can be: img_{W}_f{SS}_ch{C}.tif
	// Here, {W} stands for the well name (e.g. B02), {SS} for the position of the tile 
	// in the spiral grid (e.g. 120),
	// and {C} for the channel number (e.g. 1).
	// The length of these characters (SS and C) are the MINIMAL length of the numbers.
	//
	// Example: if the field numbers are like 00, 01, ..., 10, ..., 114, then you should enter {SS}.
	// If the field numbers are 000, 001, ..., 114, then you should enter {SSS}.
	//---------------------------------------------

	// Start with inserting the well
	
	format_split = split(filename_format, "{}");
	file_name = "";
	for (ii = 0; ii < format_split.length; ii++) {
		filename_part = format_split[ii];
		check_well = split(filename_part, "W");
		check_field = split(filename_part, "S");
		check_ch = split(filename_part, "C");
		if (check_well.length==0) { // there was a "W" between the curly brackets
			file_name = file_name + well;
		}
		else if (check_field.length==0) { // there was an arbritary number of "S"s between the curly brackets
			min_length = lengthOf(filename_part); // get the number of S's.
			file_name = file_name + number_to_string(img_nr, min_length);
		}
		else if (check_ch.length==0) { // there was an arbritary number of "c"s between the curly brackets
			min_length = lengthOf(filename_part); // get the number of c's.
			file_name = file_name + number_to_string(ch, min_length);
		}
		else { // there was no well, field or channel info in this part
			file_name = file_name + filename_part;
		}
	}
	return file_name;
}

function get_true_indeces_as_string(boolean_list){
	//---------------------------------------------
	// This function takes as input a boolean list.
	// Example: boolean_list = [true,false,true].
	// It outputs a string with the indeces where 
	// the boolean_list is True.
	// Example: boolean_list_str = "0,2".
	// If all entries are false, the function returns "None".
	//---------------------------------------------
	boolean_list_str = "";
	for (i = 0; i < boolean_list.length; i++) {
		if (boolean_list[i] == true){
			if (boolean_list_str == ""){
				boolean_list_str = boolean_list_str + d2s(i, 0);
			}
			else{
				boolean_list_str = boolean_list_str + ", " + d2s(i, 0);
			}
		}
	}
	if (boolean_list_str == ""){
		boolean_list_str = "None";
	}
	return boolean_list_str;
}

//---------------------------END FUNCTIONS-----------------------------

//---------------------------START SCRIPT------------------------------

// Get first specifications (directories, wells, number of cannels)
#@ File (label="Raw folder", style="directory") raw
#@ File (label="Root folder", style="directory") root
#@ String (label="Wells you want to process (separated by commas)") well_string
#@ String (label="Channels you want to process (separated by commas)") channel_str
#@ String (label="Filename format, e.g. img_{W}_f{SS}_ch{C}.tif (enter MFGTMP to use the standard HCA format)") filename_format
#@ int (label="Width of fused image (in number of tiles)") w
#@ int (label="Height of fused image (in number of tiles)") h

q = File.separator;

// Make a list of the channels to process
channel_str_split = split(channel_str, ",");
channels = newArray(channel_str_split.length);
for (i = 0; i < channel_str_split.length; i++) {
	channels[i] = parseInt(channel_str_split[i]);
}

// Check inputs and create directories-----------------------------------------

well_list = split(well_string, ",");
for (l = 0; l < well_list.length; l++) {

	close("*");
	
	well = well_list[l];
	input_folder = raw + q + well;
	well_folder = root + q + well;
	tile_folder = well_folder + q + "tiles";
	threshold_folder = tile_folder + q + "thresholds";
	file_list = getFileList(input_folder);

	// Check if the input folder exists
	if(!File.exists(input_folder)){
		exit("Sorry, well "+well+" does not appear in the raw folder you specified.");
	}

	// Check if the inputted channels exist
	for (c = 0; c < channels.length; c++) {
		if (filename_format=="MFGTMP"){
			file0 = find_name_of_first_image_in_list(file_list);
			first_field = input_folder + q + get_mfgtmp_file_name(file0, well, 0, channels[c]);
		}
		else {
			first_field = input_folder + q + get_file_name_from_format(filename_format, well, 0, channels[c]);
		}
		if (!File.exists(first_field)){
			exit("Sorry, I was trying to find an image with the name\n" + first_field + "\nbut I could not find it.\nDid you enter the correct channels and the correct filename format?");
		}
	}

	// Check if the inputted width/height (w) is correct
	last_file_nr = w*h - 1;
	if (filename_format=="MFGTMP"){
		file0 = find_name_of_first_image_in_list(file_list);
		last_file = input_folder + q + get_mfgtmp_file_name(file0, well, last_file_nr, channels[0]);
		one_file_too_far = input_folder + q + get_mfgtmp_file_name(file0, well, last_file_nr+1, channels[0]);
	}
	else{
		last_file = input_folder + q + get_file_name_from_format(filename_format, well, last_file_nr, channels[0]);
		one_file_too_far = input_folder + q + get_file_name_from_format(filename_format, well, last_file_nr+1, channels[0]);
	}
	if (!File.exists(last_file)){
		exit("Sorry, the chosen width ("+d2s(w,0)+") or height ("+d2s(h,0)+") cannot be correct. \nThere are less than "+d2s(w*h,0)+" images in the input folder of well "+well+".");
	}
	if (File.exists(one_file_too_far)){
		exit("Sorry, the chosen width ("+d2s(w,0)+") or height ("+d2s(h,0)+") cannot be correct. \nThere are more than "+d2s(w*h,0)+" images in the input folder of well "+well+".");
	}
	
	// Create well_folder if it did not exist already.
	if (!(File.isDirectory(well_folder))){
		File.makeDirectory(well_folder);
		print("Created a new folder for well "+well+".");
	}
	// Create tile_folder if it did not exist already.
	// If it did exist, ask the user if they want to overwrite it.
	if (!(File.isDirectory(tile_folder))){
		File.makeDirectory(tile_folder);
	}
	else{
		showMessageWithCancel("Tiles already created!","Tiles for well "+well+" were already created.\nDo you want to continue and overwrite the old tiles?");
	}
	if (!(File.isDirectory(threshold_folder))){
		File.makeDirectory(threshold_folder);
	}
}

// Get additional specifications with dialog---------------------------

// Labels and defaults
nr_channels = channels.length;
ch_labels = newArray(nr_channels);
ch_defaults = newArray(nr_channels);
for (i = 0; i < nr_channels; i++) {
	ch = channels[i];
	ch_labels[i] = "channel " + d2s(ch,0);
	ch_defaults[i] = false;
}
bit_labels = newArray("8-bit", "16-bit", "32-bit");
clockwise_labels = newArray("Clockwise", "Counter-clockwise");

Dialog.create("Specify parameters");

center_inset = 375;
Dialog.setInsets(10, 0, 0);
Dialog.addMessage("General specifications:");
Dialog.addChoice("Spiral direction (use macro clockOrAnticlock.ijm to find this out)", clockwise_labels);
Dialog.addChoice("How many bits in output?", bit_labels);
Dialog.addNumber("Downscale factor (1=no downscaling)", 1);
Dialog.setInsets(0, center_inset, 0);
Dialog.addCheckbox("Do you want to display the images?", false);

Dialog.setInsets(40, 0, 0);
Dialog.addMessage("On which channels do you want to enhance contrast? (See Process > Enhance contrast)");
Dialog.setInsets(0, center_inset, 0);
Dialog.addCheckboxGroup(1, nr_channels, ch_labels, ch_defaults);
Dialog.addNumber("Percentage of saturated pixels", 0.1);

Dialog.setInsets(40, 0, 0);
Dialog.addMessage("On which of these channels do you want to equalize histogram? (See Process > Enhance contrast)");
Dialog.setInsets(0, center_inset, 0);
Dialog.addCheckboxGroup(1, nr_channels, ch_labels, ch_defaults);

Dialog.setInsets(40, 0, 0);
Dialog.addMessage("On which channels do you want to subtract background? (See Process > Subtract background)");
Dialog.setInsets(0, center_inset, 0);
Dialog.addCheckboxGroup(1, nr_channels, ch_labels, ch_defaults);
Dialog.addNumber("Rolling ball radius", 200);

Dialog.setInsets(40, 0, 0);
Dialog.addMessage("On which channels do you want to set a threshold?");
Dialog.setInsets(0, center_inset, 0);
Dialog.addCheckboxGroup(1, nr_channels, ch_labels, ch_defaults);
Dialog.addString("Thresholding methods for these channels (start with capital letter)", "-");
Dialog.addNumber("Sigma of Gaussian blur (for thresholding)", 2);

Dialog.show();

spiral_direction = Dialog.getChoice();
nr_bits_string = Dialog.getChoice();
downscale_factor = Dialog.getNumber();
see = Dialog.getCheckbox();
ch_enhance = get_next_checkbox_group(nr_channels);
fraction_saturated = Dialog.getNumber();
ch_equalize = get_next_checkbox_group(nr_channels);
ch_subtract_background = get_next_checkbox_group(nr_channels);
rolling_ball_radius = Dialog.getNumber();
ch_threshold = get_next_checkbox_group(nr_channels);
threshold_methods = Dialog.getString();
sigma = Dialog.getNumber();
sigma_string = d2s(sigma,1);

// Check whether the number of threshold methods provided is correct.
threshold_method_list = split(threshold_methods, ",");
num_threshold_channels = 0;
for (ch = 0; ch < nr_channels; ch++) { // Count the number of channels to threshold on
	num_threshold_channels = num_threshold_channels + ch_threshold[ch];
}

if ((threshold_method_list[0] == "-") && num_threshold_channels != 0) {
	exit("Sorry, you entered invalid thresholding methods.\n \nExample:\n number of channels = 2\n threshold methods = Li,Otsu");
}
if ((threshold_method_list.length != num_threshold_channels) && (threshold_method_list[0] != "-")) {
	exit("Sorry, the number of threshold methods you provided is not the same \nas the number of channels to threshold.\n \nExample:\n number of channels = 2\n threshold methods = Li,Otsu");
}

// Convert boolean lists to string (for metadata file)-----------------------------------
ch_enhance_str = get_true_indeces_as_string(ch_enhance);
ch_equalize_str = get_true_indeces_as_string(ch_equalize);
ch_subtract_background_str = get_true_indeces_as_string(ch_subtract_background);
ch_threshold_str = get_true_indeces_as_string(ch_threshold);

// Follows-------------------------------------------------------------------------------
if (spiral_direction=="Clockwise") {
	clockwise = true;
}
else {
	clockwise = false;
}

if (!(see)) {
	setBatchMode(true);
}


// Make spiral and column grid arrays
// Note that 2D arrays are not supported by Fiji macro language, so the arrays are 1D.
// If you index it them with matrix[x + y * width], they are effectively 2D.
column_grid = make_column_grid(w,h);
spiral_grid = make_spiral_grid(w,h,clockwise);

// Loop over wells-----------------------------------------------------------------------

for (l = 0; l < well_list.length; l++) {

	// Get well names and paths
	well = well_list[l];
	input_folder = raw + q + well;
	well_folder = root + q + well;
	tile_folder = well_folder + q + "tiles";
	threshold_folder = tile_folder + q + "thresholds";

	// Save parameters to txt file
	metadata = "RawDataFolder = " + input_folder + "\n";
	metadata = metadata + "NumberOfTiles = " + d2s(w*h,0) + "\n";
	metadata = metadata + "ChannelsToProcess = " + channel_str + "\n";
	metadata = metadata + "SpiralDirection = " + spiral_direction + "\n";
	metadata = metadata + "NumberOfBits = " + nr_bits_string + "\n";
	metadata = metadata + "DownscaleFactor = " + d2s(downscale_factor,0) + "\n";
	metadata = metadata + "ChannelsEnhanceContrast = " + ch_enhance_str + "\n";
	if (ch_enhance_str != "None") {
		metadata = metadata + "FractionSaturatedPixels = " + d2s(fraction_saturated,1) + "\n";
	}
	metadata = metadata + "ChannelsEqualizeHistogram = " + ch_equalize_str + "\n";
	metadata = metadata + "ChannelsSubtractBackground = " + ch_subtract_background_str + "\n";
	if (ch_subtract_background_str != "None") {
		metadata = metadata + "RollingBallRadius = " + d2s(rolling_ball_radius,0) + "\n";
	}
	metadata = metadata + "ChannelsThreshold = " + ch_threshold_str + "\n";
	if (ch_threshold_str != "None") {
		metadata = metadata + "ThresholdMethods = " + threshold_methods + "\n";
		metadata = metadata + "SigmaGaussianBlur = " + sigma_string + "\n";
	}
	metadata_file_path = well_folder + q + well + "_parameters_prepareTiles.txt";
	File.saveString(metadata, metadata_file_path);
	print(">>>> Saved metadata file in " + well_folder + ".\n");
	
	// List all images in the raw input folder:
	file_list = getFileList(input_folder);
	
	// Get the name of the first MFGTMP file (it serves as a template).
	file0 = ""; // init
	if (filename_format=="MFGTMP") {
		file0 = find_name_of_first_image_in_list(file_list);
	}
	
	// Initialize positions on grid
	x = 0;
	y = 0;
	
	// Loop over tiles in the grid
	for (i = 0; i < w*h; i++) {
		print("Well "+well+": Processing tile "+d2s(i+1,0)+" out of "+d2s(w*h,0)+".");
	
		// Give the image the correct tile number.
		// This number is based on the column grid.
		tile_nr = column_grid[x + y*w]; // column index corresponding with (x,y) position
		img_nr = spiral_grid[x + y*w];  // spiral index corresponding with (x,y) position
		tile_nr_string = number_to_string(tile_nr,3); // for output (three numbers)
		tile_name = "tile_" + tile_nr_string;
		
		// Initialize arrays with filenames and channel numbers.
		// They will later be used to merge the channels.
		file_names = newArray(nr_channels);
		ch_counts = newArray(nr_channels);
	
		threshold_counter = 0;
		ch_th_string = "";
		
		// Loop over the channels in the image
		for (j = 0; j < nr_channels; j++){
			ch = channels[j];
			file_name = ""; // init
			if (filename_format=="MFGTMP") {
				file_name = get_mfgtmp_file_name(file0, well, img_nr, ch);
			}
			else{
				file_name = get_file_name_from_format(filename_format, well, img_nr, ch);
			}
			file_names[j] = file_name;
			ch_counts[j] = "c" + d2s(ch+1,0);
			
			// Open image
			open(input_folder + q + file_name);
			run(nr_bits_string);

			// Downsize if the user asked for it
			if(downscale_factor != 1){
				getDimensions(width, height, channels, slices, frames);
				newWidth = d2s( round(width / downscale_factor), 0 );
				newHeight = d2s( round(height / downscale_factor), 0 );
				run("Size...", "width="+newWidth+" height="+newHeight+" depth=1 constrain average interpolation=Bilinear");
			}
	
			// Perform the operations if the user asked for it:
			if ((ch_enhance[j] == true) && (ch_equalize[j] == false)){
				run("Enhance Contrast...", "saturated="+d2s(fraction_saturated,1)+" normalize");
			}
			if ((ch_enhance[j] == true) && (ch_equalize[j] == true)){
				run("Enhance Contrast...", "saturated="+d2s(fraction_saturated,1)+" normalize equalize");
			}
			if (ch_subtract_background[j] == true){
				run("Subtract Background...", "rolling="+d2s(rolling_ball_radius,0)+" disable");
			}

			if (ch_threshold[j] == true){
				// Specify channel string name for merging threshold channels
				ch_th_string = ch_th_string + "c" + d2s(threshold_counter+1,0) + "=th_" + d2s(threshold_counter+1,0) + " ";
				th_method = threshold_method_list[threshold_counter];
				run("Duplicate...", " ");
				rename("th_" + d2s(threshold_counter+1,0));
				run("Gaussian Blur...", "sigma="+sigma);
				setAutoThreshold(th_method + " dark");
				setOption("BlackBackground", true);
				run("Convert to Mask");
				threshold_counter = threshold_counter + 1;
				// Save and close thresholded image
				save(threshold_folder + q + tile_name + "_ch" + d2s(ch,0) + ".tif");
				close();
			}

			save(tile_folder + q + tile_name + "_ch" + d2s(ch,0) + ".tif");
			close();
		}

		// Update position
		y = y + 1;
		if (y == h){
			y = 0;
			x = x + 1;
		}
		
	}
	print(">>>> Finished the preparation, you can now start stitching well "+well+".");

}
print(">>>> Prepared all wells.");