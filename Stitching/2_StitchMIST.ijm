//-----------------------------------------------------------------------
// STITCH A GRID OF IMAGES USING MIST
// Macro for stitching tiles (created by high content microscope) into a fused image.
// Created 09-03-2021 by Lukas van den Heuvel.
//
// What this macro does (in chronological order):
// (1) Ask the user for experiment directory (=root), well, and number of stitched images on one axis (w).
// (2) Perform grid stitching and save the result.;
// (4) Stitch the thesholded images, if the user wants to. 
//
//
//---------------------------START FUNCTIONS-----------------------------

function number_to_string(nr, three_numbers){
	//------------------------------------------------------
	// This function converts an integer nr into a string.
	// Examples: 0 --> "00", 3 --> "03", 11 --> "11", 102 --> "102".
	// If three_numbers == true:
	// Examples: 0 --> "000", 3 --> "003", 11 --> "011", 102 --> "102".
	//------------------------------------------------------
	
	if ((nr < 10) && three_numbers){
		nr_string = "00" + d2s(nr, 0);
	}
	else if ((nr < 10) && !three_numbers){
		nr_string = "0" + d2s(nr, 0);
	}
	else if ((nr > 9) && (nr < 100) && three_numbers){
		nr_string = "0" + d2s(nr, 0);
	}
	else{
		nr_string = d2s(nr, 0);
	}
	return nr_string;
}

function find_channels_in_tile_folder(file_list){
	//------------------------------------------------------
	// This function finds which channels are present in a
	// list of tiles.
	// Example: if tile_000_ch0.tif and tile_000_ch1.tif are present
	// in file_list, then channels_present = [0,1]
	//------------------------------------------------------
	channels_present = newArray(0);
	for (i = 0; i < file_list.length; i++) {
		file = file_list[i];
		if (indexOf(file, "tile_000") == 0){
			split_file = split(file, "_ch");
			split_file = split(split_file[2], ".");
			ch_nr = parseInt( split_file[0] );
			channels_present = Array.concat(channels_present,ch_nr);
		}
	}
	return channels_present;
}

function check_if_tile_nr_is_present(nr,file_list){
	//------------------------------------------------------
	// This function checks whether a tile nr is present in
	// a list of files.
	// Example: if nr=33 and tile_033_ch*.tif is in file_list,
	// then the output is present=true.
	//------------------------------------------------------
	present = false;
	tile_name = "tile_" + number_to_string(nr, true);
	for (i = 0; i < file_list.length; i++) {
		file = file_list[i];
		if (indexOf(file, tile_name) == 0){
			present = true;
			break;
		}
	}
	return present;
}

function get_next_checkbox_group(nr_channels){
	//--------------------------------------------
	// This function outputs a boolean array with
	// checkbox choices of a checkbox group.
	//--------------------------------------------
	bool = newArray(nr_channels);
	for (i=0; i<nr_channels; i++){
		bool[i] = Dialog.getCheckbox();
	}
	return bool;
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
//---------------------------------END FUNCTIONS---------------------------------

//---------------------------------START SCRIPT----------------------------------

// Get first specifications (directories, wells, number of cannels)
#@ File (label="Path to Fiji.app", style="directory") fiji_path
#@ File (label="Root folder", style="directory") root
#@ String (label="Well you want to stitch") well
#@ int (label="Width/height of fused image") w

close("*");
setBatchMode(true);
q = File.separator;

// Follows ----------------------------------------------------------------------
well_folder = root + q + well;
tile_folder = well_folder + q + "tiles";
threshold_folder = tile_folder + q + "thresholds";
output_file = well_folder + q + well + "_fused.tif";

// Check inputs -----------------------------------------------------------------
if (!File.isDirectory(tile_folder)){
	exit("Sorry, No tile folder could be found in "+well_folder+".\nPlease check the root folder you entered.");
}
file_list = getFileList(tile_folder);
if (!check_if_tile_nr_is_present(w*w-1,file_list)){
	exit("Sorry, the chosen width/height of "+d2s(w,0)+" cannot be correct. \nThere are less than "+d2s(w*w,0)+" images in the input folder of well "+well+".");
}
if (check_if_tile_nr_is_present(w*w,file_list)){
	exit("Sorry, the chosen width/height of "+d2s(w,0)+" cannot be correct. \nThere are more than "+d2s(w*w,0)+" images in the input folder of well "+well+".");
outline_color}
if (File.exists(output_file)){
	showMessageWithCancel("WARNING","Warning: fused image is already created!\nDo you want to continue and overwrite the old fused image?");
}

// Check which channels we have -------------------------------------------------
file_list = getFileList(tile_folder);
channels = find_channels_in_tile_folder(file_list);

// Check which threshold channels we have
th_file_list = getFileList(threshold_folder);
th_channels = find_channels_in_tile_folder(th_file_list);

// Labels and defaults for dialog ------------------------------------------------
nr_channels = channels.length;
ch_labels = newArray(nr_channels);
ch_defaults = newArray(nr_channels);
default_channel_to_stitch = "channel 0";
for (i = 0; i < nr_channels; i++) {
	ch = channels[i];
	ch_labels[i] = "channel " + d2s(ch,0);
	ch_defaults[i] = true;
	if (ch==1){
		default_channel_to_stitch = "channel 1";
	}
}

nr_th_channels = th_channels.length;
th_ch_labels = newArray(nr_th_channels);
th_ch_defaults = newArray(nr_th_channels);

for (i = 0; i < nr_th_channels; i++) {
	ch = th_channels[i];
	th_ch_labels[i] = "channel " + d2s(ch,0);
	th_ch_defaults[i] = true;
}

// Dialog -------------------------------------------------------------------------
Dialog.create("Specify parameters");
Dialog.setInsets(10, 5, 0);
Dialog.addMessage("Which channels do you want to stitch?");
Dialog.setInsets(0, 330, 0);
Dialog.addCheckboxGroup(1, nr_channels, ch_labels, ch_defaults);
// Show which thresholded channels there are, if there are any
if (nr_th_channels > 0){
	Dialog.setInsets(40, 5, 0);
	Dialog.addMessage("Which thresholded channels do you want to stitch?");
	Dialog.setInsets(0, 330, 0);
	Dialog.addCheckboxGroup(1, nr_th_channels, th_ch_labels, th_ch_defaults);
	Dialog.setInsets(20, 0, 0);
	Dialog.addString("Names of thresholded channels (separated by commas)", "dapi,phalloidin");
}

ch_labels = Array.concat(ch_labels,"Brute-force (no overlap)");
Dialog.setInsets(40, 0, 0);
Dialog.addChoice("Which channel do you want to use for calculating overlap?", ch_labels, default_channel_to_stitch);
Dialog.addNumber("Estimated overlap (%)", 2);
Dialog.addNumber("Estimated overlap uncertainty (%)", 1);

Dialog.show();

// Get dialog outputs -------------------------------------------------------------------

stitch_channels = get_next_checkbox_group(nr_channels);
if (nr_th_channels > 0){ // if there are images in the threshold folder
	stitch_th_channels = get_next_checkbox_group(nr_th_channels);
	suggested_names_str = Dialog.getString();
	suggested_names = split(suggested_names_str, ",");
	// Check if the number of suggested threshold names matches the number of selected channels
	nr_selected_th_channels = 0;
	for (i = 0; i < stitch_th_channels.length; i++) {
		nr_selected_th_channels = nr_selected_th_channels + stitch_th_channels[i];
	}
	if (suggested_names_str=="-"){
		if (nr_selected_th_channels != 0) {
			exit("Sorry, you selected thresholded channels without suggesting a name for them.");
		}
	}
	else{
		if (nr_selected_th_channels != suggested_names.length) {
			exit("Sorry, the number of selected threshold channels (" + d2s(nr_selected_th_channels,0) + ") \ndoes not match the given number of thresholded names (" + suggested_names_str + ").");
		}
	}
}
ol_channel_str = Dialog.getChoice();
ol = Dialog.getNumber();
ol_uncertainty = Dialog.getNumber();

// Get overlap channel as int ------------------------------------------------------------
if(ol_channel_str == "Brute-force (no overlap)"){
	brute_force = true;
}
else{
	brute_force = false;
	split_ch = split(ol_channel_str, " ");
	ol_channel = parseInt( split_ch[1] );
}

// Get channels to stitch as array -------------------------------------------------------
channels_to_stitch = newArray(0);
first_channel = true;
ol_channel_in_channels_to_stitch = false;
for (i = 0; i < nr_channels; i++) {
	if (stitch_channels[i] == true){
		channels_to_stitch = Array.concat(channels_to_stitch,channels[i]);
		// If brute-force, make the overlap channel the first channel to stitch
		if(first_channel && brute_force){
			ol_channel = channels[i];
			first_channel = false;
		}
		if (channels[i] == ol_channel){
			ol_channel_in_channels_to_stitch = true;
		}
	}
}

// Give an error if the overlap channel is not part of the channels to stitch
if (ol_channel_in_channels_to_stitch == false){
	exit("Sorry, the channel you use to calculate overlap must be part of the channels to be stitched.")
}

// Get threshold channels to stich as array and as string----------------------------------
th_channels_to_stitch = newArray(0);
for (i = 0; i < nr_th_channels; i++) {
	if (stitch_th_channels[i] == true){
		th_channels_to_stitch = Array.concat(th_channels_to_stitch,th_channels[i]);
	}
}

// Save parameters to metadata file
stitch_channels_str = get_true_indeces_as_string(stitch_channels);
if (nr_th_channels > 0){
	stitch_th_channels_str = get_true_indeces_as_string(stitch_th_channels);
}
else{
	stitch_th_channels_str = "None";
}
metadata = "NumberOfFields = " + d2s(w*w,0) + "\n";
metadata = metadata + "ChannelsStitched = " + stitch_channels_str + "\n";
metadata = metadata + "ThresholdedChannelsStitched = " + stitch_th_channels_str + "\n";
if (stitch_th_channels_str != "None"){
	metadata = metadata + "NamesOfThresholdedChannels = " + suggested_names_str + "\n";
}
metadata = metadata + "ChannelUsedForCalculatingOverlap = " + ol_channel_str + "\n";
metadata = metadata + "EstimatedOverlap = " + d2s(ol,1) + "%\n";
metadata = metadata + "OverlapUncertainty = " + d2s(ol_uncertainty,1) + "%\n";
metadata_file_path = well_folder + q + well + "_parameters_stitching.txt";
File.saveString(metadata, metadata_file_path);
print(">>>> Saved metadata file in " + well_folder + ".\n");

// Stitch overlap channel -----------------------------------------------------------------

if (brute_force) {
	print("\n>>> BRUTE-FORCE STITCH ON CHANNEL "+d2s(ol_channel,0)+"...\n");
	run("MIST", "gridwidth="+d2s(w,0)+" gridheight="+d2s(w,0)+" starttile=0 imagedir="+tile_folder+" filenamepattern=tile_{ppp}_ch{t}.tif filenamepatterntype=SEQUENTIAL gridorigin=UL assemblefrommetadata=false assemblenooverlap=true globalpositionsfile=[] numberingpattern=VERTICALCOMBING startrow=0 startcol=0 extentwidth="+d2s(w,0)+" extentheight="+d2s(w,0)+" timeslices="+d2s(ol_channel,0)+" istimeslicesenabled=true outputpath="+well_folder+" displaystitching=true outputfullimage=false outputmeta=true outputimgpyramid=false blendingmode=LINEAR blendingalpha=NaN outfileprefix=img- programtype=AUTO numcputhreads=16 loadfftwplan=true savefftwplan=true fftwplantype=MEASURE fftwlibraryname=libfftw3 fftwlibraryfilename=libfftw3.dll planpath=["+fiji_path+q+"lib"+q+"fftw"+q+"fftPlans] fftwlibrarypath=["+fiji_path+q+"lib"+q+"fftw] stagerepeatability=0 horizontaloverlap=0 verticaloverlap=0 numfftpeaks=0 overlapuncertainty=0 isusedoubleprecision=false isusebioformats=false issuppressmodelwarningdialog=false isenablecudaexceptions=false translationrefinementmethod=SINGLE_HILL_CLIMB numtranslationrefinementstartpoints=16 headless=false loglevel=MANDATORY debuglevel=NONE");
}
else{
	print("\n>>> COMPUTING OVERLAP OF TILES OF CHANNEL "+d2s(ol_channel,0)+"...\n");
	run("MIST", "gridwidth="+d2s(w,0)+" gridheight="+d2s(w,0)+" starttile=0 imagedir="+tile_folder+" filenamepattern=tile_{ppp}_ch{t}.tif filenamepatterntype=SEQUENTIAL gridorigin=UL assemblefrommetadata=false assemblenooverlap=false globalpositionsfile=[] numberingpattern=VERTICALCOMBING startrow=0 startcol=0 extentwidth="+d2s(w,0)+" extentheight="+d2s(w,0)+" timeslices="+d2s(ol_channel,0)+" istimeslicesenabled=true outputpath="+well_folder+" displaystitching=true outputfullimage=false outputmeta=true outputimgpyramid=false blendingmode=LINEAR blendingalpha=NaN outfileprefix=img- programtype=AUTO numcputhreads=16 loadfftwplan=true savefftwplan=true fftwplantype=MEASURE fftwlibraryname=libfftw3 fftwlibraryfilename=libfftw3.dll planpath=["+fiji_path+q+"lib"+q+"fftw"+q+"fftPlans] fftwlibrarypath=["+fiji_path+q+"lib"+q+"fftw] stagerepeatability=0 horizontaloverlap="+d2s(ol,0)+" verticaloverlap="+d2s(ol,0)+" numfftpeaks=0 overlapuncertainty="+d2s(ol_uncertainty,0)+" isusedoubleprecision=false isusebioformats=false issuppressmodelwarningdialog=false isenablecudaexceptions=false translationrefinementmethod=SINGLE_HILL_CLIMB numtranslationrefinementstartpoints=16 headless=false loglevel=MANDATORY debuglevel=NONE");
}
rename("Fused-c"+d2s(ol_channel,0));

// Stitch other channels, if they are present ---------------------------------------------

// Path to output file with tile positions
positions_file = well_folder + q + "img-global-positions-"+d2s(ol_channel,0)+".txt";
remaining_channels = Array.deleteValue(channels_to_stitch, ol_channel);

if (remaining_channels.length > 0){
	
	print("\n>>> STITCHING REMAINING CHANNELS...\n");
	for (i = 0; i < remaining_channels.length; i++) {
		ch = remaining_channels[i];
		positions_file_copy = well_folder + q + "img-global-positions-"+d2s(ch,0)+".txt";
		File.copy(positions_file, positions_file_copy);
	
		// Rempove these two files, to avoid an annoying warning message
		if (File.exists(well_folder + q + "img-log.txt")){
			File.delete(well_folder + q + "img-log.txt");
		}
		if (File.exists(well_folder + q + "img-statistics.txt")){
			File.delete(well_folder + q + "img-statistics.txt");
		}
	
		run("MIST", "gridwidth="+d2s(w,0)+" gridheight="+d2s(w,0)+" starttile=0 imagedir="+tile_folder+" filenamepattern=tile_{ppp}_ch{t}.tif filenamepatterntype=SEQUENTIAL gridorigin=UL assemblefrommetadata=true assemblenooverlap=false globalpositionsfile="+well_folder+q+"img-global-positions-{t}.txt numberingpattern=VERTICALCOMBING startrow=0 startcol=0 extentwidth="+d2s(w,0)+" extentheight="+d2s(w,0)+" timeslices="+d2s(ch,0)+" istimeslicesenabled=true outputpath="+well_folder+" displaystitching=true outputfullimage=false outputmeta=false outputimgpyramid=false blendingmode=LINEAR blendingalpha=NaN outfileprefix=img- programtype=AUTO numcputhreads=16 loadfftwplan=true savefftwplan=true fftwplantype=MEASURE fftwlibraryname=libfftw3 fftwlibraryfilename=libfftw3.dll planpath=["+fiji_path+q+"lib"+q+"fftw"+q+"fftPlans] fftwlibrarypath="+fiji_path+q+"lib"+q+"fftw] stagerepeatability=0 horizontaloverlap="+d2s(ol,0)+" verticaloverlap="+d2s(ol,0)+" numfftpeaks=0 overlapuncertainty="+d2s(ol_uncertainty,0)+" isusedoubleprecision=false isusebioformats=false issuppressmodelwarningdialog=false isenablecudaexceptions=false translationrefinementmethod=SINGLE_HILL_CLIMB numtranslationrefinementstartpoints=16 headless=false loglevel=MANDATORY debuglevel=NONE");
		rename("Fused-c"+d2s(ch,0));
	}
	
	// Find the right color for right channel (reverse order)
	channel_string = "";
	for (i = 0; i < channels_to_stitch.length; i++){
		ch = channels_to_stitch[i];
		ch_count = "c" + d2s(ch+1,0);
		channel_string = channel_string + ch_count + "=" + "Fused-c" + d2s(channels_to_stitch.length - ch - 1,0) + " ";
	}
	// Merge the channels
	run("Merge Channels...", channel_string + "create");
}

// Show result ---------------------------------------------------------------------------
rename(well+"_fused");
setSlice(nSlices);
run("Blue");
setBatchMode("exit and display");

// Ask the user if they are happy with the thresholds
choice = newArray("Yes", "No");
Dialog.createNonBlocking("Thresholded result");
Dialog.addChoice("Are you happy with the fused image?", choice, "Yes");
Dialog.show();
s = Dialog.getChoice();
if(s == "No"){
	exit("Macro was aborted by user");
}

// Ask the user to change LUT
title1 = "Set the right colors.";
message1 = "Change the LUT of the channels to set the right colors.\nPress OK when you are done.";
waitForUser(title1, message1);

// Make oval selection
if (roiManager("count") > 0){
	roiManager("Deselect");
	roiManager("Delete");
}
setTool("oval");

title2 = "Make oval selection";
message2 = "\nFit a circular selection to the well borders while holding shift.\nPress OK when you are done.";
waitForUser(title2, message2);

roiManager("Add");

// Crop image
run("Crop");
run("Clear Outside");

// Save result (both as seperate channels and as RGB color) --------------------------------
print("Saving the result...");
saveAs("Tiff", output_file);

print("Converting to 8-bit RGB and saving it as seperate tif tile...");
run("RGB Color");
saveAs(root+q+well+q+well+"_fused_RGB.tif");
close("*");

// Stitch thresholds -----------------------------------------------------------------------
if (th_channels_to_stitch.length > 0){
	
	print("\n>>> STITCHING THRESHOLDED CHANNELS...\n");
	for (i = 0; i < th_channels_to_stitch.length; i++) {
		ch = th_channels_to_stitch[i];
		// Copy global positions file (if it didn't already exist)
		positions_file_copy = well_folder + q + "img-global-positions-"+d2s(ch,0)+".txt";
		if (!(File.exists(positions_file_copy))){
			File.copy(positions_file, positions_file_copy);
		}
	
		// Remove these two files, to avoid an annoying warning message
		if (File.exists(well_folder + q + "img-log.txt")){
			File.delete(well_folder + q + "img-log.txt");
		}
		if (File.exists(well_folder + q + "img-statistics.txt")){
			File.delete(well_folder + q + "img-statistics.txt");
		}
		
		run("MIST", "gridwidth="+d2s(w,0)+" gridheight="+d2s(w,0)+" starttile=0 imagedir="+threshold_folder+" filenamepattern=tile_{ppp}_ch{t}.tif filenamepatterntype=SEQUENTIAL gridorigin=UL assemblefrommetadata=true assemblenooverlap=false globalpositionsfile="+well_folder+q+"img-global-positions-{t}.txt numberingpattern=VERTICALCOMBING startrow=0 startcol=0 extentwidth="+d2s(w,0)+" extentheight="+d2s(w,0)+" timeslices="+d2s(ch,0)+" istimeslicesenabled=true outputpath="+well_folder+" displaystitching=true outputfullimage=false outputmeta=false outputimgpyramid=false blendingmode=LINEAR blendingalpha=NaN outfileprefix=img- programtype=AUTO numcputhreads=16 loadfftwplan=true savefftwplan=true fftwplantype=MEASURE fftwlibraryname=libfftw3 fftwlibraryfilename=libfftw3.dll planpath=["+fiji_path+q+"lib"+q+"fftw"+q+"fftPlans] fftwlibrarypath=["+fiji_path+q+"lib"+q+"fftw] stagerepeatability=0 horizontaloverlap="+d2s(ol,0)+" verticaloverlap="+d2s(ol,0)+" numfftpeaks=0 overlapuncertainty="+d2s(ol_uncertainty,0)+" isusedoubleprecision=false isusebioformats=false issuppressmodelwarningdialog=false isenablecudaexceptions=false translationrefinementmethod=SINGLE_HILL_CLIMB numtranslationrefinementstartpoints=16 headless=false loglevel=MANDATORY debuglevel=NONE");
		rename(well+"_th_"+suggested_names[i]);

		// Ask the user if they are happy with the thresholds
		choice = newArray("Yes", "No");
		Dialog.createNonBlocking("Thresholded result");
		Dialog.addChoice("Are you happy with the thresholds?", choice, "Yes");
		Dialog.show();
		s = Dialog.getChoice();

		// If yes, save result
		if(s=="Yes"){
			// Crop thresholded image
			ch_name = getString("Name of this channel:", suggested_names[i]);
			roiManager("Select", 0);
			run("Crop");
			run("Clear Outside");
			print("Saving the thresholded result...");
			save(well_folder + q + well+"_th_"+ ch_name +".tif");
			close();
		}

		// If not, let the user see the tile nrs.
		else{

			W = getWidth();
			H = getHeight();
			width = round(W / w);
			height = round(H / w);
			setJustification("center");
			setFont("SansSerif", 300);
			setColor("white");

			column_grid = make_column_grid(w,w);
			x = 0;
			y = 0;
			
			for (i = 0; i < w*w; i++) {
				tile_nr = column_grid[x + y*w]; // column index corresponding with (x,y) position
				drawString(tile_nr, x*width + width/2, y*height + height/2, "black");
				// Update position
				y = y + 1;
				if (y == w){
					y = 0;
					x = x + 1;
				}
			}
		
			exit("These are the tile numbers. Use the macro thresholdIndividualTiles to manually set a threshold on the tiles you wish to adjust.");
		}
	}
}

print("\n>>> SUCCESSFULLY STITCHED WELL " + well);


