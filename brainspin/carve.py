"""
Copyright 2023 Netherlands eScience Center and University of Amsterdam.
Licensed under <TBA>. See LICENSE for details.

This file contains methods to anonymize image files.
Specifically we have a methods to get the image out of DICOMs,
and to strip soft tissues off head MRIs.

Note the package has an optional pydicom dependancy, without it this module
has functions related to dicoms that will not work.
"""
# import libraries


import os
import glob
from abc import ABC, abstractmethod

import numpy
from datetime import datetime, date
import pydicom as dicom
import pandas as pd
import skimage.io as io

from pydicom.multival import MultiValue
from pydicom.sequence import Sequence


class Source(ABC):
    """
    This class is provided as a helper for those who want to implement
    their own sources.
    It is not necessary to extend this class.  Our code doesn't yet do
    type checking, but if you want to ensure type checking on your side,
    you may inherit from this class.
    """

    @abstractmethod
    def get_tag(self):
        """
        The value returned from this function should be suitable for
        pandas to name a column.
        """
        raise NotImplementedError()

    @abstractmethod
    def items(self, reader, transformer=None):
        """
        This function will be expected to produce file names or file-like
        objects of DICOM files.  The results will be then fed to either
        pydicom or SimpleITK libraries for metadata extraction.
        This function should return a generator yielding a tuple of two
        elements.  First element will be inserted into the source column
        (the one labeled by :code:`get_tag()` method), the second is the
        result of calling :code:`reader`.
        :param reader: A function that takes an individual source, either
                       a file path or a file-like object, and returns the
                       processed metadata.
        :param transformer: Optionally, the caller of this function will
                            supply a transformer function that needs to
                            be called on the value that will be stored
                            in the source column of the resulting DataFrame
        """
        while False:
            yield None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Source:
            get_tag_found, items_found = False, False
            for sub in C.mro():
                for prop in sub.__dict__:
                    if prop == 'get_tag':
                        get_tag_found = True
                    elif prop == 'items':
                        items_found = True
                    if get_tag_found and items_found:
                        return True
        return NotImplemented


class DirectorySource:
    """Class to aid reading DICOMs, package agnostically"""

    def __init__(self, directory, tag):
        self.directory = directory
        self.tag = tag

    def get_tag(self):
        return self.tag

    def items(self, reader, transformer=None):
        for file in os.listdir(self.directory):
            full_path = os.path.join(self.directory, file)
            parsed = reader(full_path)
            if transformer is not None:
                full_path = transformer(full_path)
            yield full_path, parsed


class GlobSource:
    """
    Class to aid finding files from path (for later reading out DICOM)
    """

    def __init__(self, exp, tag, recursive=True):
        self.exp = exp
        self.tag = tag
        self.recursive = recursive

    def get_tag(self):
        return self.tag

    def items(self, reader, transformer=None):
        for file in glob(self.exp, recursive=self.recursive):
            parsed = reader(file)
            if transformer is not None:
                full_path = transformer(file)
            yield file, parsed


class MultiSource:

    def __init__(self, tag, *sources):
        self.tag = tag
        self.sources = sources

    def get_tag(self):
        return self.tag

    def items(self, reader, transformer=None):
        for s in self.sources:
            for key, parsed in s.items(reader, transformer):
                yield key, parsed


def rename_file(original, target, ext):
    dst_file = os.path.basename(original)
    dst_file = os.path.splitext(dst_file)[0]
    return os.path.join(target, '{}.{}'.format(dst_file, ext))


class PydicomDicomReader:
    """Class for reading DICOM metadata with pydicom."""

    exclude_field_types = (Sequence, MultiValue, bytes)
    """
    Default types of fields not to be included in the dataframe
    produced from parsed DICOM files.
    """

    date_fields = ('ContentDate', 'SeriesDate', 'ContentDate', 'StudyDate')
    """
    Default DICOM tags that should be interpreted as containing date
    information.
    """

    time_fields = ('ContentTime', 'StudyTime')
    """
    Default DICOM tags that should be interpreted as containing
    datetime information.
    """

    exclude_fields = ()
    """
    Default tags to be excluded from genrated :code:`DataFrame` for any
    other reason.
    """

    def __init__(
            self,
            exclude_field_types=None,
            date_fields=None,
            time_fields=None,
            exclude_fields=None,
    ):
        """
        Initializes the reader with some filtering options.
        :param exclude_field_types: Some DICOM types have internal structure
                                    difficult to represent in a dataframe.
                                    These are filtered by default:
                                    * :class:`~pydicom.sequence.Sequence`
                                    * :class:`~pydicom.multival.MultiValue`
                                    * :class:`bytes` (this is usually the
                                      image data)
        :type exclude_field_types: Sequence[type]
        :param date_fields: Fields that should be interpreted as having
                            date information in them.
        :type date_fields: Sequence[str]

        :param time_fields: Fields that should be interpreted as having
                            time information in them.
        :type time_fields: Sequence[str]
        :param exclude_fields: Fields to exclude (in addition to those selected
                               by :code:`exclude_field_types`
        :type exclude_fields: Sequence[str]
        """
        if exclude_field_types:
            self.exclude_field_types = exclude_field_types
        if date_fields:
            self.date_fields = date_fields
        if exclude_fields:
            self.exclude_fields = exclude_fields

    def dicom_date_to_date(self, source):
        """
        Utility method to help translate DICOM dates to :class:`~datetime.date`
        :param source: Date stored as a string in DICOM file.
        :type source: str
        :return: Python date object.
        :rtype: :class:`~datetime.date`
        """
        year = int(source[:4])
        month = int(source[4:6])
        day = int(source[6:])
        return date(year=year, month=month, day=day)

    def rip_out_jpgs(self, source, destination):
        """
        Extract image data from DICOM files and save it as JPG in
        :code:`destination`.
        :param source: A source generator.  For extended explanation see
                       :class:`~carve.Source`.
        :type source: :class:`~carve.Source`
        :param destination: The name of the directory where JPG files
                            should be stored.
        :type destination: Compatible with :func:`os.path.join`
        """
        for key, parsed in source.items(dicom.dcmread):
            io.imsave(
                rename_file(key, destination, 'jpg'),
                parsed.pixel_array,
            )

    def read(self, source):
        """
        This function allows reading of metadata in what source gives.
        :param source: A source generator.  For extended explanation see
                       :class:`~carve.Source`.
        :type source: :class:`~carve.Source`
        :return: dataframe with metadata from dicoms
        :rtype: :class:`~pandas.DataFrame`
        """

        tag = source.get_tag()
        columns = {tag: []}
        colnames = set([])
        excluded_columns = set([])
        for key, parsed in source.items(dicom.dcmread):
            for field in parsed.dir():
                colnames.add(field)
                val = parsed[field].value
                if isinstance(val, self.exclude_field_types):
                    excluded_columns.add(field)
        colnames -= excluded_columns
        colnames -= set(self.exclude_fields)
        for key, parsed in source.items(dicom.dcmread, os.path.basename):
            columns[tag].append(key)
            for field in colnames:
                val = parsed[field].value
                col = columns.get(field, [])
                if field in self.date_fields:
                    val = self.dicom_date_to_date(val)
                # elif field in self.time_fields:
                #     val = self.dicom_time_to_time(val)
                elif isinstance(val, int):
                    val = int(val)
                elif isinstance(val, float):
                    val = float(val)
                elif isinstance(val, str):
                    val = str(val)
                col.append(val)
                columns[field] = col
        return pd.DataFrame(columns)


def get_numpy_with_pydicom(dicom_folder_path, numpy_folder_path):
    """
    This function is for users with pydicom library only.
    If you do not have the library it will throw an error.
    The function function jpeg files out of a dicom file directory,
    one by one, each of them (not just the first series as), and puts them in
    an out put directory.
    :param dicom_folder_path: dicomfile_directory, directory with dicom/.dcm
    :type dicom_folder_path: str
    :param jpg_folder_path: output_directory, where they should be placed
    :type jpg_folder_path: str
    :return: lovely (will put your images in the new folder but not return them)
    :rtype: bool
    """
    images_path = os.listdir(dicom_folder_path)
    for n, image in enumerate(images_path):
        ds = dicom.dcmread(os.path.join(dicom_folder_path, image))
        pixel_array_numpy = ds.pixel_array
        image = image.replace('.dcm', '.npy')
        lovely = io.imsave(
            os.path.join(numpy_folder_path, image),
            pixel_array_numpy,
        )

    print('{} image converted'.format(n))
    return lovely


def get_jpg_with_pydicom(dicom_folder_path, jpg_folder_path):
    """
    This function is for users with pydicom library only.
    If you do not have the library it will throw an error.
    The function function jpeg files out of a dicom file directory,
    one by one, each of them (not just the first series as), and puts them in
    an out put directory.
    :param dicom_folder_path: dicomfile_directory, directory with dicom/.dcm
    :type dicom_folder_path: str
    :param jpg_folder_path: output_directory, where they should be placed
    :type jpg_folder_path: str
    :return: love (will put your images in the new folder but not return them)
    :rtype: bool
    """
    images_path = os.listdir(dicom_folder_path)
    for n, image in enumerate(images_path):
        ds = dicom.dcmread(os.path.join(dicom_folder_path, image))
        pixel_array_numpy = ds.pixel_array
        image = image.replace('.dcm', '.jpg')
        love = io.imsave(
            os.path.join(jpg_folder_path, image),
            pixel_array_numpy,
        )

    print('{} image converted'.format(n))
    return love


def rename_file(original, target, ext):
    dst_file = os.path.basename(original)
    dst_file = os.path.splitext(dst_file)[0]
    return os.path.join(target, '{}.{}'.format(dst_file, ext))
