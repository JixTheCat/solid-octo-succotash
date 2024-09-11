#from numpy import astype
from os import path, makedirs
from sqlalchemy import create_engine, text

import logging
import pandas as pd


class SWA:
    def __init__(self, pwd: str, populate=False):
        logging.getLogger().setLevel(logging.INFO)

        #TODO
        # The connection silently fails
        logging.info('Creating new connection...')
        self.cnx = self.new_connection(pwd)
        #logging.info('Connection established!')

        missing = 0

        if path.exists('swa'):
            logging.info('Previous files were found!')
            if path.exists(path.join('swa', 'data_years.feather')):
                self.data_years = pd.read_feather(
                    path.join('swa', 'data_years.feather')
                )
                logging.info('\tloaded data_years')
            else:
                missing += 1


            if path.exists(path.join('swa', 'fertilisers.feather')):
                self.fertilisers = pd.read_feather(
                    path.join('swa', 'fertilisers.feather')
                )
                logging.info('\tloaded fertilisers')
            else:
                missing += 1

            if path.exists(path.join('swa', 'fields.feather')):
                self.fields = pd.read_feather(
                    path.join('swa', 'fields.feather')
                )
                logging.info('\tloaded fields')
            else:
                missing += 1

            if path.exists(path.join('swa', 'field_options.feather')):
                self.field_options = pd.read_feather(
                    path.join('swa', 'field_options.feather')
                )
                logging.info('\tloaded field_options')
            else:
                missing += 1

            if path.exists(path.join('swa', 'gi_regions.feather')):
                self.gi_regions = pd.read_feather(
                    path.join('swa', 'gi_regions.feather')
                )
                logging.info('\tloaded gi_regions')
            else:
                missing += 1

            if path.exists(path.join('swa', 'gi_zones.feather')):
                self.gi_zones = pd.read_feather(
                    path.join('swa', 'gi_zones.feather')
                )
                logging.info('\tloaded gi_regions')
            else:
                missing += 1

        # if anything is missing and we want to populate we do!
        if populate and (missing != 0):
            self.populate()

    @staticmethod
    def new_connection(pwd: str):
        """Create a new connection the SWA database.

        Args:
             pwd: The password to connect to the database.

        Returns:
             A MySQL connector.
        """
        return create_engine("mysql+mysqlconnector://" +
                             "swa:" +
                             "{}".format(pwd) +
                             "@172.105.187.76/swa",
                             pool_size=1,
                             max_overflow=0)

    def query(self, select: str):
        """execute a query using the class object's connection.

        Returns:
            A pandas dataframe of the returned view.
        """
        with self.cnx.connect() as conn:
            answer = pd.read_sql_query(text(select), conn)
        return answer

    def create(self, populate=True, limit=0):
        """
        create a feather file of all the answers in the swa DB.
        :return:
        """

        logging.info('Downloading data...')

        if populate:
            logging.info('Populating data structures...')
            self.populate()

        if limit > 0:
             answers = self.query(
                 'select * from answers LIMIT {}'.format(limit))
        else:
            answers = self.query('select * from answers')

        answers.to_feather(path.join('swa', 'answers.feather'))

        answers['field'] = answers['field_id'].copy()

        answers['field'].replace(
            dict(zip(self.fields['id'], self.fields['name'])),
            inplace=True)
        answers['data_year_id'].replace(
            dict(
                zip(self.data_years['id'], self.data_years['label'])),
            inplace=True)

        # The magic number 13 is the field ID for gi_region
        answers[answers['field'] == 13]['value'].astype(
            int).replace(
            dict(
                zip(self.gi_regions['id'], self.gi_regions['label'])),
            inplace=True)

        # The magic number 12 is the field ID for gi_zone
        answers[answers['field'] == 12]['value'].astype(
            int).replace(
            dict(
                zip(self.gi_zones['id'], self.gi_zones['label'])),
            inplace=True)

        answers[answers['field'].isin(
            self.field_options['id'])]['value'].astype(
            int).replace(
            dict(zip(
                self.field_options['id'],
                self.field_options['label'])),
            inplace=True)

        # The magic numbers listed are the field IDs for fertilisers
        answers[answers['field'].isin(
            [195, 197, 199, 201, 203,
             205, 207, 209, 211, 213])]['value'].astype(
            int).replace(
            dict(zip(
                self.fertilisers['id'],
                self.fertilisers['label'])),
            inplace=True)

        answers.to_feather(path.join('swa', 'answers.feather'))

    def populate(self):
        if not path.exists('swa'):
            makedirs('swa')
        else:
            logging.warning("Folder already exists. Files will" +
                            " be overwritten!!!")

        self.data_years = self.query(
            'select id, label from data_years')
        self.data_years.to_feather(
            path.join('swa', 'data_years.feather')
        )

        self.fertilisers = self.query(
            'select id, label from fertilisers')
        self.fertilisers.to_feather(
            path.join('swa', 'fertilisers.feather')
        )

        self.fields = self.query(
            'select id, name from fields')
        self.fields.to_feather(
            path.join('swa', 'fields.feather')
        )

        self.field_options = self.query(
            'select id, field_id label from field_options')
        self.fields.to_feather(
            path.join('swa', 'field_options.feather')
        )

        self.gi_regions = self.query(
            'select id, label from gi_regions')
        self.gi_regions.to_feather(
            path.join('swa', 'gi_regions.feather')
        )

        self.gi_zones = self.query(
            'select id, label from gi_zones')
        self.gi_zones.to_feather(
            path.join('swa', 'gi_zones.feather')
        )

    def treasury_api(self):
        """Returns member IDs who sell to Treasury

        This function is primarily to remember which MySQL table
            contains information regarding sales to treasury.
            Notably there is also a table that lists the accesses
            made to the API as well through the table: api_accesses
        """
        return self.query("select * from api_access_member")

    def table_list(self):
        """Lists tables currently present in the database"""
        return self.query("SHOW TABLES")
