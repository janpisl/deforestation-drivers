
import ee
from geetools import cloud_mask

import pdb

class Landsat_cloud_processor():
    def __init__(self) -> None:
        pass

    def get_cloudfree_coll(self, start_date, end_date, clip_geometry, landsat_version, aoi, max_cloud):
                

        if landsat_version == 7:
            collection_name = f"LANDSAT/LE07/C02/T1_L2"
        elif landsat_version == 8:
            collection_name = 'LANDSAT/LC08/C02/T1_L2'
        else:
            raise NotImplementedError(f"Landsat '{landsat_version}' not supported. Implemented Landsat versions are 7 and 8")

        # Import and filter 
        collection = (ee.ImageCollection(collection_name)
            .filterBounds(aoi)
            .filterDate(start_date, end_date))\
            .filter(ee.Filter.lt('CLOUD_COVER', int(max_cloud)))\
            .map(lambda image: image.clipToCollection(clip_geometry))

        collection = collection.select(['SR_B1', 'SR_B2' ,'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'],['B1', 'B2', 'B3', 'B4', 'B5', 'pixel_qa'])

        metadata = collection.getInfo()
        print(len(metadata['features']))

        clean_coll = collection.map(self.clean)

        return clean_coll

    def clean(self, image):
        """This is from geetools 
        """
        
        mask = cloud_mask.landsatSR(['cloud', 'shadow', 'adjacent'])

        return mask(image)

        

'''//Function to mask clouds based on the pixel_qa band of Landsat 8 SR data.
function maskL8sr(image) {
  // Bits 3 and 5 are cloud shadow and cloud, respectively.
  var cloudShadowBitMask = (1 << 3);
  var cloudsBitMask = (1 << 5);
  // Get the pixel QA band.
  var qa = image.select('pixel_qa');
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
                 .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  return image.updateMask(mask);
}'''





