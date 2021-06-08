//-----------------------------------------------------------------------
// CHOOSE CLOCKWISE OR ANTICLOCKWISE ROTAION
// Created 09-03-2021 by Lukas van den Heuvel.
//
// For some unknown reason, the HCA sometimes spirals clockwise and sometimes
// anticlockwise when imaging a well.
// This macro shows you the first 4 tiles in a spiral, both in clockwise and
// anticlockwise direction, s.t. you can easily see the direction.
//-----------------------------------------------------------------------

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

function get_img_file_name(file0, well, nr, ch){
	
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
	nr_string = number_to_string(nr,false);
	file_name = parts[0] + "_" + parts[1] + "_" + well + "f" + nr_string + "d" + d2s(ch,0) + ".TIF";
	return file_name;
}

function find_name_of_first_image_in_list(file_list){
	// Some files in a raw directory might not be images.
	// This function finds the name of the first HCA image in the directory.
	// This name serves as a template to find other files.
	for (i = 0; i < file_list.length; i++) {
		fileName = file_list[i];
		splitFileName = split(fileName, "_");
		if (splitFileName[0] == "MFGTMP") { // The file is an image made by the HCA
			img0 = fileName;
			break
		}
	}
	print(i);
	return img0;
}


// User input
#@ File (label="Raw folder", style="directory") raw
#@ String (label="Well") well
#@ int (label="Channel to display") channel

q = File.separator;
close("*");

input_folder = raw + q + well;
// List all images in the raw input folder:
file_list = getFileList(input_folder);
// Get the name of the first image (it serves as a template).
file0  = find_name_of_first_image_in_list(file_list);
print(file0);


img0 = get_img_file_name(file0, well, 0, channel);
img1 = get_img_file_name(file0, well, 1, channel);
img2 = get_img_file_name(file0, well, 2, channel);
img3 = get_img_file_name(file0, well, 3, channel);

width = round( screenWidth / 5 );
height = width;

// Open 4 images clockwise

open(input_folder + q + img0);
setLocation(0, 0, width, height);
run("Enhance Contrast...", "saturated=0.1");

open(input_folder + q + img1);
setLocation(width, 0, width, height);
run("Enhance Contrast...", "saturated=0.1");

open(input_folder + q + img2);
setLocation(width, height, width, height);
run("Enhance Contrast...", "saturated=0.1");

open(input_folder + q + img3);
setLocation(0, height, width, height);
run("Enhance Contrast...", "saturated=0.1");

// Open 4 images counterclockwise

halfScreen = round( screenWidth / 2 );

open(input_folder + q + img3);
setLocation(halfScreen, 0, width, height);
run("Enhance Contrast...", "saturated=0.1");

open(input_folder + q + img2);
setLocation(halfScreen+width, 0, width, height);
run("Enhance Contrast...", "saturated=0.1");

open(input_folder + q + img1);
setLocation(halfScreen+width, height, width, height);
run("Enhance Contrast...", "saturated=0.1");

open(input_folder + q + img0);
setLocation(halfScreen, height, width, height);
run("Enhance Contrast...", "saturated=0.1");

// Show a message
choices = newArray("Left", "Right");
Dialog.createNonBlocking("Clockwise or counterclockwise?"); 
Dialog.addMessage("LEFT: clockwise.");
Dialog.addMessage("RIGHT: counterclockwise.");
Dialog.addMessage("Check which option is correct and write it down.");
Dialog.addChoice("Option", choices)
Dialog.setLocation(round(screenWidth/2 - width),round(screenHeight/2)); 
Dialog.show();

lr = Dialog.getChoice();
if (lr=="Left"){
	result = "clockwise";
}
else if (lr=="Right"){
	result = "counterclockwise";
}
print("Chosen spiralling direction in well "+well+": "+result);
