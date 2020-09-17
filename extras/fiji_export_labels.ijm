// Get user input
print("Please select the directory with labeled images...");
pathin = getDirectory("Choose directory");

// Create output directory
pathout = pathin + "labels/";
File.makeDirectory(pathout);

// Get file lists
filelist = getFileList(pathin);
print("Found " + lengthOf(filelist) + " files");

for (i = 0; i < lengthOf(filelist); i++) {
    if (endsWith(filelist[i], ".tif")) { 
		print("Processing: " + filelist[i]);
    	
    	// Open images
        open(pathin + filelist[i]);

		// Convert Fiji's strange units into pixel units
		run("Set Scale...", "distance=0 known=0 pixel=1 unit=pixel");

        // Generate a list of spots
        run("Measure");

		// Save output to file
		bname = split(filelist[i], ".");
		fnameout = pathout + bname[0] + ".csv";
		saveAs("Results", fnameout);

		// Close opened windows
		selectWindow("Results"); 
        run("Close" );
		selectImage(nImages());  
		run("Close");
    } 
}

print("Processing complete!");