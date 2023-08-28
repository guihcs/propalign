# PropMatch: Property Matcher

PropMatch is a Python 3.9-based ontology property matching system designed to find better alignment of properties across different ontologies. This system uses lexical matching methods and alignment extension combined with different embeddings to increase the amount of correspondences found between properties.

## Download

A packaged version of PropMatch is available for download [here](https://drive.google.com/file/d/1UShYKSO8fle-VWC4o1YZ2xxsgVVyELZ4/view?usp=drive_link). It follows the MELT Web API protocol packaged in a Docker container.

## Development

PropMatch was tested on Python 3.9. To run PropMatch you also need to download the Finnish word embeddings from
http://dl.turkunlp.org/finnish-embeddings/finnish_4B_parsebank_skgram.bin.

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions to PropMatch are welcome! If you encounter issues or have suggestions for improvements, please feel free to open an issue or submit a pull request in the [PropMatch GitHub repository](https://github.com/guihcs/propalign).

## License

PropMatch is released under the [MIT License](https://opensource.org/licenses/MIT).

---

For inquiries and support, contact us at Guilherme.Santos-Sousa@irit.fr.