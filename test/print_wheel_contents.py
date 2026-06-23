print("!!!!!!!!!! Printing wheel contents !!!!!!!!!!")

import glob
import os
import sys


def list_package(pkgname):
    site_packages_dirs = [
        p for p in sys.path if os.path.isdir(p) and "site-packages" in p
    ]

    found = False
    for sp in site_packages_dirs:
        pkg_candidates = glob.glob(os.path.join(sp, f"{pkgname}*"))
        for pkg in pkg_candidates:
            print(f"Contents of installed package/module at {pkg}:")
            if os.path.isdir(pkg):
                for dp, dn, filenames in os.walk(pkg):
                    for f in filenames:
                        print(os.path.join(dp, f))
            else:
                # Compiled extension (pyd/so)
                print(pkg)
            found = True

    if not found:
        print("Cannot find stablebear in site-packages")


list_package("stablebear")
list_package("stablebear_cpu")
list_package("_sb_cpp")
list_package("stablebear-cpu")
