// Get specifications 
#@ File (label="Input MFGTMP folder", style="directory") input_path
#@ File (label="Output folder", style="directory") output_path
#@ String (label="Well name") well
#@ int (label="Timelapse channel") ch
#@ int (label="DAPI channel") ch_dapi
#@ int (label="SIR-Actin channel") ch_actin
#@ boolean (label="Do you want to align timeframes using StackReg?") stackreg
#@ String (label="Name of timelapse channel") ch_name
#@ String (label="Agonist added") agonist
#@ String (label="Concentration in uM") conc_str
#@ String (label="Volume added in uL") vol_str
#@ String (label="Start time of measurement (in seconds after ATP addition)") startT_str
#@ boolean (label="Do you want to save metadata?") save_metadata

close("*");
q = File.separator;
setBatchMode(true);

// Convert strings to decimal numbers
conc = parseFloat(conc_str);
vol = parseFloat(vol_str);
startT = parseFloat(startT_str);

// Get MFGTMP name (name of input directory)
split_path = split(input_path,q);
mfgtmp_name = split_path[split_path.length-1];  // File name is last entry of path

// Init paths to files
tif_file_path = input_path + q + well;
kinetics_path = input_path + q + mfgtmp_name + "_kineticprotocol.log";
metadata_path = output_path + q + well + "_Ca_metadata.txt";
timelapse_file =  well + "_" + ch_name + "_timelapse.tif";
timelapse_path = output_path + q + timelapse_file;
background_path = output_path + q + well +"_"+ ch_name + "_backgound_intensity.tif";

// check inputs
if (!File.isDirectory(tif_file_path)) {
	exit("Sorry, there is no folder with TIFF images in the input MFGTMP folder.\nCheck whether you entered the correct folder.")
}
if (!File.exists(kinetics_path)) {
	exit("Sorry, the metadata file " + mfgtmp_name + "_kineticprotocol.log is missing in " + input_path + ".\nCheck the input MFGTMP folder.")
}
if (File.exists(metadata_path) && save_metadata) {
	showMessageWithCancel("Metadata already created!","WARNING: A metadata file with the name "+ well + "_metadata.txt already exists in the output folder.\nDo you want to continue and overwrite the existing metadata file?");
}
if (!File.exists(metadata_path) && !save_metadata) {
	showMessageWithCancel("No metadata file available","WARNING: No metadata file exists yet for well "+well+", and you chose not to create one./nDo you want to continue without creating a metadata file?");
}
if (File.exists(timelapse_path)) {
	showMessageWithCancel("Stack already created!","WARNING: An image stack file with the name "+ well + "_timelapse_"+ch_name+".tif already exists in the output folder.\nDo you want to continue and overwrite the existing stack?");
}
showMessageWithCancel("Check inputs","You entered the following parameters:\n\nWell = "+well+"\nAgonist = "+agonist+"\n[Agonist] = "+d2s(conc,3)+" uM\nVolume agonist = "+d2s(vol,3)+" uL\nStart time = "+d2s(startT,0)+" s.\n\nPress OK to continue.");

// Get first input TIF file
file_list = getFileList(tif_file_path);
for (i = 0; i < file_list.length; i++) {
	file_name = file_list[i];
	if (indexOf(file_name, "d"+d2s(ch,0)) > 0){
		first_file_path = tif_file_path + q + file_name;
		file_name_split1 = split(file_name, "f");
		file_name_split2 = split(file_name_split1[1], "d");
		field_nr_str = file_name_split2[0];
		break;
	}
}

// Read file with kinetics protocol
kinetics_file = File.openAsString(kinetics_path);
kinetics_lines = split(kinetics_file, "\n");

// Get metadata from kinetics protocol
metadata = "RawDataFolder = " + input_path + "\n";
metadata = metadata + "Field = " + field_nr_str + "\n";
for (i = 0; i < kinetics_lines.length; i++) {
	split_line = split(kinetics_lines[i], "=");
	if (split_line.length > 0) {
		if (split_line[0] == "Name"){
			metadata = metadata + "ProtocolName = " + split_line[1] + "\n";
		}
		else if (split_line[0] == "TotalKineticsExecutionTime") {
			metadata = metadata + "DurationOfMeasurement = " + split_line[1] + " s\n";
		}
		else if (split_line[0] == "MinimumScanInterval") {
			metadata = metadata + "MinimumScanInterval = " + split_line[1] + " s\n";
		}
	}
}

// Add ATP concentration and start time of measurement to metadata file
metadata = metadata + "Agonist = " + agonist + "\n";
metadata = metadata + "[Agonist] = " + d2s(conc,3) + " uM\n";
metadata = metadata + "VolumeAgonist = " + d2s(vol,3) + " uL\n";
metadata = metadata + "StartTime = " + d2s(startT,0) + " s\n";

// Save metadata file
if (save_metadata) {
	File.saveString(metadata, metadata_path);	
	print("\n>>>> Saved metadata file: " + metadata_path);
}

// Load and save image sequence
print(">>>> Loading image sequence ...");
run("Image Sequence...", "open="+first_file_path+" file=d"+d2s(ch,0)+" sort");
//run("Green");
rename("Timelapse");
// Align timeframes
if (stackreg) {
	print(">>>> Performing stack registration ...");
	setSlice(1);
	run("StackReg", "transformation=Translation");
}

print(">>>> Saving image sequence ...");
saveAs("Tiff", timelapse_path);
print(">>>> Saved image stack in " + timelapse_path);

// Save network image (Hoechst + SIR-actin) -----------------------------------

// Get DAPI img
dapi_img_path = "";
actin_img_path = "";
for (i = 0; i < file_list.length; i++) {
	dapi_file_name = file_list[i];
	if (indexOf(dapi_file_name, "d"+d2s(ch_dapi,0)) > 0){
		dapi_img_path = tif_file_path + q + dapi_file_name;
		break;
	}
}
// Get SIR-actin image
for (i = 0; i < file_list.length; i++) {
	actin_file_name = file_list[i];
	if (indexOf(actin_file_name, "d"+d2s(ch_actin,0)) > 0){
		actin_img_path = tif_file_path + q + actin_file_name;
		break;
	}
}
rgb_exists = false;
rgb_file = well + "_f" + field_nr_str + "_RGB.tif";
rgb_file_path = output_path + q + rgb_file;
// If only DAPI image was found, save it
if (!(dapi_img_path=="") && (actin_img_path=="")) {
	open(dapi_img_path);
	run("Enhance Contrast...", "saturated=0.1");
	run("Blue");
	//run("RGB Color");
	saveAs("Tiff", rgb_file_path);
	rgb_exists = true;
}
// If only actin image was found, save it
else if ((dapi_img_path=="") && !(actin_img_path=="")) {
	open(actin_img_path);
	run("Enhance Contrast...", "saturated=0.1 equalize");
	run("Red");
	run("RGB Color");
	saveAs("Tiff", rgb_file_path);
	rgb_exists = true;
}
// Merge & save Hoechst and SIR actin, if they were both found
else if (!(dapi_img_path=="") && !(actin_img_path=="")) {
	open(dapi_img_path);
	run("Enhance Contrast...", "saturated=0.1");
	open(actin_img_path);
	run("Enhance Contrast...", "saturated=0.1 equalize");
	run("Merge Channels...", "c1="+actin_file_name+" c3="+dapi_file_name+" create");
	close(dapi_file_name);
	close(actin_file_name);
	run("RGB Color");
	saveAs("Tiff", rgb_file_path);
	close("Composite");
	print(">>>> Saved live network RGB image in " + rgb_file_path);
	rgb_exists = true;
}
else{
	print("WARNING: could not find the Hoechst and SIR-actin channels.");
}

// Get background intensity levels --------------------------------------------
//setBatchMode("exit and display");
//selectWindow(timelapse_file);
//getDimensions(width, height, channels, slices, frames);

// Empty ROI manager
//if (roiManager("count")>0){
//	roiManager("deselect");
//	roiManager("delete");
//}
// Initialize a selection on the RGB image if that exists,
// otherwise make the selection on the timelapse
//if (rgb_exists){
//	selectWindow(rgb_file);
//}
//d = round(width / 20); // initial diameter of selection
//m = round(width / 2);  // Center of the image
//setTool("oval");
//makeOval(m, m, d, d);

// Let the user move / change the selection, and add it to the ROI manager
//title = "Define a background selection.";
//message = "Select an area without cells.\nPress OK when you are done.";
//waitForUser(title, message);
//roiManager("Add");
//setBatchMode(true);

// Clear results table
//run("Set Measurements...", "mean redirect=None decimal=3");
//run("Clear Results");

// Get mean intensity of selection at every timepoint
//selectWindow(timelapse_file);
//background_intensity = newArray(slices);
//for (t = 0; t < slices; t++) {
//	setSlice(t+1); // slice number counting starts at 1
//	roiManager("measure");
//	background_intensity[t] = getResult("Mean", nResults-1);
//}

// Make a timelapse with the background intensity values
//nBits = bitDepth();
//newImage("Background", d2s(nBits,0)+"-bit black", width, height, slices);
//for (t = 0; t < slices; t++) {
//	setSlice(t+1); // slice number counting starts at 1
//	background_value = background_intensity[t];
//	run("Add...", "value="+d2s(background_value,0)+" slice");
//}

// Save background intensity values as timelapse
//print(">>>> Saving background intensity as timelapse ...");
//saveAs("Tiff", background_path);
//print(">>>> Saved image stack in " + background_path);

setBatchMode("exit and display");
