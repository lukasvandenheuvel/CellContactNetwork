
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

#@ File (label="Fused Image (RGB)") path_to_fused
#@ File (label="Tile") path_to_tile
#@ String (label="Well") well
#@ int (label="Tile Number") tile_nr

close("*");
root = File.getParent(path_to_fused);
q = File.separator;

// Open images
print(">>>> Opening images...");
open(path_to_fused);
rename("Fused");
getDimensions(w_fused, h_fused, ch_fused, slices, frames);
open(path_to_tile);
rename("Tile");
getDimensions(w_tile, h_tile, ch_tile, slices, frames);

// Clear LOG window
print("\\Clear");

// Do alignment
print(">>>> Aligning...");
run("Pairwise stitching", "first_image=Fused second_image=Tile fusion_method=[Overlay into composite image] fused_image=Alignment check_peaks=5 compute_overlap x="+d2s(w_fused/2,0)+" y="+d2s(h_fused/2,0)+" registration_channel_image_1=[Average all channels] registration_channel_image_2=[Average all channels]");

// Get info from LOG
logString = getInfo( "log" );

// Extract coordinates
split1 = split(logString, "( correlation)");
split2 = split(split1[0], "(second relative to first)");
split_coordinates = split(split2[1], ",");
split_x = split(split_coordinates[0], "(");
x = parseFloat(split_x[1]);
split_y = split(split_coordinates[1], ")");
y = parseFloat(split_y[0]);

// Put results in a string
result = "Well = " + well + "\n";
result = result + "TileNr = " + d2s(tile_nr,0) + "\n";
result = result + "PathToFused = " + path_to_fused + "\n";
result = result + "PathToTile = " + path_to_tile + "\n";
result = result + "TileWidth = " + d2s(w_tile,0) + "\n";
result = result + "TileHeight = " + d2s(h_tile,0) + "\n";
result = result + "TileXposition = " + d2s(x,0) + "\n";
result = result + "TileYposition = " + d2s(y,0) + "\n";

// Save results
output_file = root + q + well + "_results_aligmentTileF" + number_to_string(tile_nr,2) + ".txt";
File.saveString(result, output_file);

showMessage("Success!\nAlignment results are saved in \n"+output_file);