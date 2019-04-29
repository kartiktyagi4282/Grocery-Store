package com.example.android.minorsem5;

import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentManager;
import android.support.v4.app.FragmentPagerAdapter;

class SectionsPagerAdapter extends FragmentPagerAdapter {

    public SectionsPagerAdapter(FragmentManager fm) {
        super(fm);
    }

    @Override
    public Fragment getItem(int position) {

        switch(position){
            case 0:
                AllStoreFragment allStoreFragment = new AllStoreFragment();
                return allStoreFragment;

            case 1:
                RequestsFragment requestsFragment = new RequestsFragment();
                return requestsFragment;

            case 2:
                AccountDetailsFragment accountDetailsFragment = new AccountDetailsFragment();
                return accountDetailsFragment;

                default:
                    return null;
        }
    }

    @Override
    public int getCount() {
        return 3;
    }

    @Nullable
    @Override
    public CharSequence getPageTitle(int position) {

        switch (position){
            case 0:
                return "STORES";
            case 1:
                return "SHOPPING";
            case 2:
                return "DETAILS";
                default:
                    return null;
        }
    }
}
