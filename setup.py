from setuptools import setup


setup(
      name='dense_model_to_pipeline',
      version='1.0',
      description='''
            This is a sample python package for encapsulating custom
            tranforms from scikit-learn into Watson Machine Learning
      ''',
      url='https://github.com/joel021/create_model_to_pipeline',
      author='Vanderlei Munhoz, Joel',
      author_email='vnderlev@protonmail.ch',
      license='BSD',
      packages=[
            'transforms_package'
      ],
      zip_safe=False
)
