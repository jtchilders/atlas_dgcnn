#!/usr/bin/env python
import argparse,logging,tarfile,os
import numpy as np
logger = logging.getLogger(__name__)

DEFAULT_FRAC_TRAIN = 0.8


def main():
   ''' simple starter program that can be copied for use when starting a new script. '''
   logging_format = '%(asctime)s %(levelname)s:%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging_level = logging.INFO
   
   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-i','--input',help='input tarball',required=True)
   parser.add_argument('-o','--output',help='output path to store dataset',required=True)
   parser.add_argument('-f','--fraction_train',help='fraction of the dataset for training; the remaining is used for testing and validation [default=%04.2f]' % DEFAULT_FRAC_TRAIN, default=DEFAULT_FRAC_TRAIN,type=float)

   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename',dest='logfilename',default=None,help='if set, logging information will go to file')
   args = parser.parse_args()

   if args.debug and not args.error and not args.warning:
      logging_level = logging.DEBUG
   elif not args.debug and args.error and not args.warning:
      logging_level = logging.ERROR
   elif not args.debug and not args.error and args.warning:
      logging_level = logging.WARNING

   logging.basicConfig(level=logging_level,
                       format=logging_format,
                       datefmt=logging_datefmt,
                       filename=args.logfilename)

   datatar = tarfile.open(args.input,'r:gz')

   files = datatar.getnames()
   nfiles = len(files)
   logger.info('total files: %s',nfiles)
   logger.info('first file: %s',files[0])

   # randomize the file order to mix up the class files
   np.random.shuffle(files)
   np.random.shuffle(files)

   # need a text file that lists filenames for training and testing
   train_filelist_filename = os.path.join(args.output,'zej_train.txt')
   test_filelist_filename = os.path.join(args.output,'zej_test.txt')

   # calculate number of files in each dataset
   n_train_files = int(nfiles * args.fraction_train)
   # n_test_files = nfiles - n_train_files

   # create output dir if it doesn't exist
   if not os.path.exists(args.output):
      os.makedirs(args.output)

   # write filelist for training
   with open(train_filelist_filename,'w') as trainfile:
      for file in files[0:n_train_files]:
         if file.endswith('.csv'):
            trainfile.write(os.path.join(args.output,file) + '\n')
   
   # write filelist for testing
   with open(test_filelist_filename,'w') as testfile:
      for file in files[n_train_files:]:
         if file.endswith('.csv'):
            testfile.write(os.path.join(args.output,file) + '\n')
   
   # unpack files
   datatar.extractall(args.output)

   logger.info('''Please update your input config file `configs/atlas_dgcnn.json`
under the `data` section to include these two parameters:
"train_filelist":                "%s",
"test_filelist":                 "%s",
''',train_filelist_filename,test_filelist_filename)


if __name__ == "__main__":
   main()
