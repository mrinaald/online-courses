# README

To extract the slides from the tar splits, first join the splits using the following command:
```sh
# cat slides.tar.xz.part.a* > slides.tar.xz
```

You can check the SHA1 checksum of the joined file to make sure that the data is not corrupted. To check the SHA1 checksum, use the following command:
```sh
# sha1sum -c slides.sha1
```

If the status is "OK", then use the following command to untar the file:
```sh
# tar -xf slides.tar.xz
```

