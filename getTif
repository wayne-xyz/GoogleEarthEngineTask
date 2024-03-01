//this geometroy is manully painted
// the table is what I upload to assets , which is shapefile



// function get date range to check
var getDateRange=function(imageCollection){
  var dateRange = nicfi.reduceColumns(ee.Reducer.minMax(), ['system:time_start']);
  // Extract the start and end dates from the date range
  var startDate = ee.Date(dateRange.get('min'));
  var endDate = ee.Date(dateRange.get('max'));
  print('Start date',startDate);
  print('end date',endDate);
  return{
    startDate:startDate,
    endDate:endDate,
  }
}

// This collection is not publicly accessible. To sign up for access,
// please see https://developers.planet.com/docs/integrations/gee/nicfi
var nicfi = ee.ImageCollection('projects/planet-nicfi/assets/basemaps/americas');
//get date of origin source
var originRange=getDateRange(nicfi);

// Filter basemaps by date and get the first image from filtered results
var basemap= nicfi.filter(ee.Filter.date('2024-01-01','2025-04-01')).first(); // first will choose the early one 

// get the date range of currently basemap
var currDate=ee.Date(basemap.get('system:time_start'));
// Convert the acquisition date to human-readable format
var acquisitionDateString = currDate.format('YYYY-MM-dd');

// Print the acquisition date of the basemap
print('Basemap Acquisition Date:', acquisitionDateString);



Map.centerObject(basemap, 4);

var vis = {'bands':['R','G','B'],'min':64,'max':5454,'gamma':1.8};

Map.addLayer(basemap, vis, '2024-01 mosaic');
Map.addLayer(
    basemap.normalizedDifference(['N','R']).rename('NDVI'),
    {min:-0.55,max:0.8,palette: [
        '8bc4f9', 'c9995c', 'c7d270','8add60','097210'
    ]}, 'NDVI', false);
    


//Select certain value of certain 
var fieldName='AREA_M2';
var desiredValue=491193.839474;
var filteredTable=table.filter(ee.Filter.eq(fieldName,desiredValue));
var selectedGeometry=filteredTable.geometry();

// Add the geometry of the first feature as a layer to the map
Map.addLayer(selectedGeometry, {color: 'red'}, "First Feature Geometry");

print("selectedGeometry", selectedGeometry);
// Center the map on the geometry of the first feature
Map.centerObject(selectedGeometry, 10);

// save to currently google account drive
// ready to save  this is rectangle
Export.image.toDrive({
    image: basemap, // <--
    description: 'image',
    scale: 5, //Resolution is about 5 it will 
    region: selectedGeometry  
});

// this is cliped shape of image
var clipped =basemap.clip(selectedGeometry)
Export.image.toDrive({
    image: clipped, // <--
    description: 'clip_image',
    scale: 5, //Resolution is about 5 it will 
    region: selectedGeometry  
});
