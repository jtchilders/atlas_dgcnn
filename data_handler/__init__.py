import logging
logger = logging.getLogger('data_handler')

__all__ = ['random_file_generator','h5_file_generator','csv_file_generator','atlas_pointcloud_csv']
from . import random_file_generator,h5_file_generator,csv_file_generator
from . import atlas_pointcloud_csv


def get_datasets(config):

   if config['data']['handler'] in globals():
      logger.info('using data handler %s',config['data']['handler'])
      handler = globals()[config['data']['handler']]
   else:
      raise Exception('failed to find data handler %s in globals %s' % (config['data']['handler'],globals()))

   return handler.get_datasets(config)
