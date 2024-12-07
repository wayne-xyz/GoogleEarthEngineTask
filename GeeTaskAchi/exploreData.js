


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

//function get the data's count of the collection
var getDateLayerSize=function(imageCollection){
  print('Count of this data source layer:',imageCollection.size())
  return imageCollection.size()
}

// function get all the date 
var getallDate=function(image){
  var date=ee.Date(image.get('system:time_start'))
  return ee.Feature(null,{'date':date})
}


// This collection is not publicly accessible. To sign up for access,
// please see https://developers.planet.com/docs/integrations/gee/nicfi
var nicfi = ee.ImageCollection('projects/planet-nicfi/assets/basemaps/americas');
//get date of origin source
var originRange=getDateRange(nicfi);
//get the size of the collection
var dataSize=getDateLayerSize(nicfi);

// map the feature over the collection
var dateFeatures=nicfi.map(getallDate)
var dateList=dateFeatures.aggregate_array('date');
print('List of dates:',dateList)



// Filter basemaps by date and get the first image from filtered results
var basemap= nicfi.filter(ee.Filter.date('2024-01-01','2025-04-01')).sort('system:time_start', false).first(); // last one  will choose the early one 

// get the date range of currently basemap
var currDate=ee.Date(basemap.get('system:time_start'));
// Convert the acquisition date to human-readable format
var acquisitionDateString = currDate.format('YYYY-MM-dd');

// Print the acquisition date of the basemap
print('Basemap Acquisition Date:', acquisitionDateString);

// get the bands of the basemap
var bandNames=basemap.bandNames();
print("Band Names:",bandNames)


// show the data
Map.centerObject(basemap, 4);
var vis = {'bands':['R','G','B'],'min':64,'max':5454,'gamma':1.8};
Map.addLayer(basemap, vis, '2024-01 mosaic');
Map.addLayer(
    basemap.normalizedDifference(['N','R']).rename('NDVI'),
    {min:-0.55,max:0.8,palette: [
        '8bc4f9', 'c9995c', 'c7d270','8add60','097210'
    ]}, 'NDVI', false);
    
