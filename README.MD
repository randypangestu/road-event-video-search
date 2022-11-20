# Road Event Video Search

Repository ini menyediakan data, script dan contoh demo untuk Road Event Video Search WebApp.

## Deskripsi

Road Event Video Search adalah sebuah sistem yang menggabungkan data teks dan visual untuk mendeteksi anomali pada lalu lintas kendaraan dan pejalan kaki.

Sistem ini menggunakan model zero-shot learning yang dilatih menggunakan data visual dan teks. Karena dilatih dengan jumlah data yang sangat banyak dan beragam, model ini dapat digunakan untuk berbagai jenis aplikasi, yakni memiliki sifat yang general.

Dalam projek ini, kami akan mengaplikasikan model zero-shot learning CLIP pada kasus spesifik, yaitu fenomena lalu lintas. Kami mengelompokan fenomena lalu lintas menjadi crashes, traffic jam, flood, dan demonstration.

## Install

### Jalankan Langsung di Colab (Opsi 1)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mj8fv8NxFg_9udbfZB8vc6IbBj88rLBQ?usp=sharing)

Untuk mencoba WebApp ini bisa langsung mengunjungi link di atas untuk melakukan deployment WebApp di Colab. Jalankan cell yang ada secara berurutan dan nantinya akan muncul link Gradio yang bisa diakses secara public.

### Install Library Menggunakan pip (Opsi 2)

```bash
cd Source/WebApps/
pip3 install -r requirements.txt
```

### Jalankan Menggunakan Python

```bash
cd Source/WebApps/
python3 Quiet_Wire_WA.py
```
