//! Sample-based regression tests for cascade classifier.
//!
//! These tests use manually verified ground truth labels.
//! They ensure the classifier produces correct results on real-world torrent data.

use crate::cascade::{Cascade, Medium};
use crate::{ContentInfo, FileInfo};
use std::sync::LazyLock;

static CASCADE: LazyLock<Cascade> = LazyLock::new(|| {
    Cascade::default_with_ml().expect("Failed to create cascade")
});

#[test]
fn sample_001() {
    // Герберт Розендорфер - Четверги с прокурором (Классический де
    let info = ContentInfo {
        name: "Герберт Розендорфер - Четверги с прокурором (Классический детектив).2007".to_string(),
        files: vec![
        FileInfo {
            path: "Герберт Розендорфер - Четверги с прокурором (Классический детектив).2007.pdf".to_string(),
            filename: "Герберт Розендорфер - Четверги с прокурором (Классический детектив).2007.pdf".to_string(),
            size: 1363148,
        },
        FileInfo {
            path: "Герберт Розендорфер - Четверги с прокурором (Классический детектив).2007.fb2".to_string(),
            filename: "Герберт Розендорфер - Четверги с прокурором (Классический детектив).2007.fb2".to_string(),
            size: 1258291,
        },
        FileInfo {
            path: "Герберт Розендорфер - Четверги с прокурором (Классический детектив).2007.rtf".to_string(),
            filename: "Герберт Розендорфер - Четверги с прокурором (Классический детектив).2007.rtf".to_string(),
            size: 741580,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Book, "Sample 1: {:?}", result);
}

#[test]
fn sample_002() {
    // Безжалостные люди - Ruthless People (by alenavova).avi
    let info = ContentInfo {
        name: "Безжалостные люди - Ruthless People (by alenavova).avi".to_string(),
        files: vec![
        FileInfo {
            path: "Безжалостные люди - Ruthless People (by alenavova).avi".to_string(),
            filename: "Безжалостные люди - Ruthless People (by alenavova).avi".to_string(),
            size: 1503238553,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 2: {:?}", result);
}

#[test]
fn sample_003() {
    // ATID-505-HD
    let info = ContentInfo {
        name: "ATID-505-HD".to_string(),
        files: vec![
        FileInfo {
            path: "ATID-505-HD.mp4".to_string(),
            filename: "ATID-505-HD.mp4".to_string(),
            size: 1717986918,
        },
        FileInfo {
            path: "ATID-505-EP.mp4".to_string(),
            filename: "ATID-505-EP.mp4".to_string(),
            size: 9122611,
        },
        FileInfo {
            path: "ATID-505-C.jpg".to_string(),
            filename: "ATID-505-C.jpg".to_string(),
            size: 327475,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 3: {:?}", result);
}

#[test]
fn sample_004() {
    // Protest The Hero
    let info = ContentInfo {
        name: "Protest The Hero".to_string(),
        files: vec![
        FileInfo {
            path: "2016 - Pacific Myth (Deluxe Edition) [WEB]/06 Caravan.flac".to_string(),
            filename: "06 Caravan.flac".to_string(),
            size: 70254592,
        },
        FileInfo {
            path: "2016 - Pacific Myth (EP)/06 - Caravan.flac".to_string(),
            filename: "06 - Caravan.flac".to_string(),
            size: 69310873,
        },
        FileInfo {
            path: "2016 - Pacific Myth (Deluxe Edition) [WEB]/12 Caravan (Instrumental).flac".to_string(),
            filename: "12 Caravan (Instrumental).flac".to_string(),
            size: 68996300,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Audio, "Sample 4: {:?}", result);
}

#[test]
fn sample_005() {
    // AKB-019 - 战争女神制服诱惑
    let info = ContentInfo {
        name: "AKB-019 - 战争女神制服诱惑".to_string(),
        files: vec![
        FileInfo {
            path: "AKB-019 - 战争女神制服诱惑.mp4".to_string(),
            filename: "AKB-019 - 战争女神制服诱惑.mp4".to_string(),
            size: 1395864371,
        },
        FileInfo {
            path: "AKB-019 - 战争女神制服诱惑.jpg".to_string(),
            filename: "AKB-019 - 战争女神制服诱惑.jpg".to_string(),
            size: 305868,
        },
        FileInfo {
            path: "如何永久找到我们.jpg".to_string(),
            filename: "如何永久找到我们.jpg".to_string(),
            size: 245862,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 5: {:?}", result);
}

#[test]
fn sample_006() {
    // The.Hobbit.The.Battle.of.the.Five.Armies.2014.HUN.PAL.DVDR-N
    let info = ContentInfo {
        name: "The.Hobbit.The.Battle.of.the.Five.Armies.2014.HUN.PAL.DVDR-Netman".to_string(),
        files: vec![
        FileInfo {
            path: "The.Hobbit.The.Battle.of.the.Five.Armies-Netman.iso".to_string(),
            filename: "The.Hobbit.The.Battle.of.the.Five.Armies-Netman.iso".to_string(),
            size: 4724464025,
        },
        FileInfo {
            path: "The.Hobbit.The.Battle.of.the.Five.Armies-Netman.nfo".to_string(),
            filename: "The.Hobbit.The.Battle.of.the.Five.Armies-Netman.nfo".to_string(),
            size: 16896,
        },
        FileInfo {
            path: "The.Hobbit.The.Battle.of.the.Five.Armies-Netman.sfv".to_string(),
            filename: "The.Hobbit.The.Battle.of.the.Five.Armies-Netman.sfv".to_string(),
            size: 132,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 6: {:?}", result);
}

#[test]
fn sample_007() {
    // Castle.S08.rus.LostFilm.TV
    let info = ContentInfo {
        name: "Castle.S08.rus.LostFilm.TV".to_string(),
        files: vec![
        FileInfo {
            path: "Castle.S08E22.rus.LostFilm.TV.avi".to_string(),
            filename: "Castle.S08E22.rus.LostFilm.TV.avi".to_string(),
            size: 616248115,
        },
        FileInfo {
            path: "Castle.S08E03.rus.LostFilm.TV.avi".to_string(),
            filename: "Castle.S08E03.rus.LostFilm.TV.avi".to_string(),
            size: 531523174,
        },
        FileInfo {
            path: "Castle.S08E12.rus.LostFilm.TV.avi".to_string(),
            filename: "Castle.S08E12.rus.LostFilm.TV.avi".to_string(),
            size: 521666560,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 7: {:?}", result);
}

#[test]
fn sample_008() {
    // Half.Nelson.2006.720p.BluRay.H264.AAC-RARBG
    let info = ContentInfo {
        name: "Half.Nelson.2006.720p.BluRay.H264.AAC-RARBG".to_string(),
        files: vec![
        FileInfo {
            path: "Half.Nelson.2006.720p.BluRay.H264.AAC-RARBG.mp4".to_string(),
            filename: "Half.Nelson.2006.720p.BluRay.H264.AAC-RARBG.mp4".to_string(),
            size: 1395864371,
        },
        FileInfo {
            path: "RARBG.txt".to_string(),
            filename: "RARBG.txt".to_string(),
            size: 30,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 8: {:?}", result);
}

#[test]
fn sample_009() {
    // OnlyFans.2023.Thestartofus.Happy.Cowgirl.Sunday.XXX.1080p.MP
    let info = ContentInfo {
        name: "OnlyFans.2023.Thestartofus.Happy.Cowgirl.Sunday.XXX.1080p.MP4-XXX[XC]".to_string(),
        files: vec![
        FileInfo {
            path: "onlyfans.2023.thestartofus.happy.cowgirl.sunday.xxx.mp4".to_string(),
            filename: "onlyfans.2023.thestartofus.happy.cowgirl.sunday.xxx.mp4".to_string(),
            size: 246205644,
        },
        FileInfo {
            path: "Torrent Downloaded From XXXClub.to .nfo".to_string(),
            filename: "Torrent Downloaded From XXXClub.to .nfo".to_string(),
            size: 34,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 9: {:?}", result);
}

#[test]
fn sample_010() {
    // [HardKinks.com] Macho Beast (Jessy Ares, Mario Domenech).mp4
    let info = ContentInfo {
        name: "[HardKinks.com] Macho Beast (Jessy Ares, Mario Domenech).mp4".to_string(),
        files: vec![
        FileInfo {
            path: "[HardKinks.com] Macho Beast (Jessy Ares, Mario Domenech).mp4".to_string(),
            filename: "[HardKinks.com] Macho Beast (Jessy Ares, Mario Domenech).mp4".to_string(),
            size: 882900992,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 10: {:?}", result);
}

#[test]
fn sample_011() {
    // Brickfoot (2014) - When I'm Gone [mp3-320]
    let info = ContentInfo {
        name: "Brickfoot (2014) - When I'm Gone [mp3-320]".to_string(),
        files: vec![
        FileInfo {
            path: "15 - Everything Must Have an End.mp3".to_string(),
            filename: "15 - Everything Must Have an End.mp3".to_string(),
            size: 17301504,
        },
        FileInfo {
            path: "05 - The Guy Who Comes in Last.mp3".to_string(),
            filename: "05 - The Guy Who Comes in Last.mp3".to_string(),
            size: 16567500,
        },
        FileInfo {
            path: "01 - Happy (When I'm Gone).mp3".to_string(),
            filename: "01 - Happy (When I'm Gone).mp3".to_string(),
            size: 14260633,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Audio, "Sample 11: {:?}", result);
}

#[test]
fn sample_012() {
    // Экзистенция-1999 (BDRip_720p).mkv
    let info = ContentInfo {
        name: "Экзистенция-1999 (BDRip_720p).mkv".to_string(),
        files: vec![
        FileInfo {
            path: "Экзистенция-1999 (BDRip_720p).mkv".to_string(),
            filename: "Экзистенция-1999 (BDRip_720p).mkv".to_string(),
            size: 4294967296,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 12: {:?}", result);
}

#[test]
fn sample_013() {
    // SlavicPunk-Oldtimer-v29.06.2023
    let info = ContentInfo {
        name: "SlavicPunk-Oldtimer-v29.06.2023".to_string(),
        files: vec![
        FileInfo {
            path: "SlavicPunk-Oldtimer-v29.06.2023_setup-2.bin".to_string(),
            filename: "SlavicPunk-Oldtimer-v29.06.2023_setup-2.bin".to_string(),
            size: 2040109465,
        },
        FileInfo {
            path: "SlavicPunk-Oldtimer-v29.06.2023_setup-3.bin".to_string(),
            filename: "SlavicPunk-Oldtimer-v29.06.2023_setup-3.bin".to_string(),
            size: 2040109465,
        },
        FileInfo {
            path: "SlavicPunk-Oldtimer-v29.06.2023_setup-4.bin".to_string(),
            filename: "SlavicPunk-Oldtimer-v29.06.2023_setup-4.bin".to_string(),
            size: 2040109465,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Software, "Sample 13: {:?}", result);
}

#[test]
fn sample_014() {
    // Hazbin Hotel - Live on Broadway (2025).1080p.H265.EAC3.6CH-M
    let info = ContentInfo {
        name: "Hazbin Hotel - Live on Broadway (2025).1080p.H265.EAC3.6CH-MNKYDDL.mkv".to_string(),
        files: vec![
        FileInfo {
            path: "Hazbin Hotel - Live on Broadway (2025).1080p.H265.EAC3.6CH-MNKYDDL.mkv".to_string(),
            filename: "Hazbin Hotel - Live on Broadway (2025).1080p.H265.EAC3.6CH-MNKYDDL.mkv".to_string(),
            size: 1932735283,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 14: {:?}", result);
}

#[test]
fn sample_015() {
    // Squirt Monsters 2 - 1080p
    let info = ContentInfo {
        name: "Squirt Monsters 2 - 1080p".to_string(),
        files: vec![
        FileInfo {
            path: "Squirt Monsters 2.mp4".to_string(),
            filename: "Squirt Monsters 2.mp4".to_string(),
            size: 4939212390,
        },
        FileInfo {
            path: "Squirt Monsters 2.mp4.jpg".to_string(),
            filename: "Squirt Monsters 2.mp4.jpg".to_string(),
            size: 586649,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 15: {:?}", result);
}

#[test]
fn sample_016() {
    // 01 Let Me Go (feat. Chad Kroeger).mp3
    let info = ContentInfo {
        name: "01 Let Me Go (feat. Chad Kroeger).mp3".to_string(),
        files: vec![
        FileInfo {
            path: "01 Let Me Go (feat. Chad Kroeger).mp3".to_string(),
            filename: "01 Let Me Go (feat. Chad Kroeger).mp3".to_string(),
            size: 10800332,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Audio, "Sample 16: {:?}", result);
}

#[test]
fn sample_017() {
    // avav77.xyz@YMDD312
    let info = ContentInfo {
        name: "avav77.xyz@YMDD312".to_string(),
        files: vec![
        FileInfo {
            path: "avav77.xyz@YMDD312.mp4".to_string(),
            filename: "avav77.xyz@YMDD312.mp4".to_string(),
            size: 5046586572,
        },
        FileInfo {
            path: "人间尤物.mp4".to_string(),
            filename: "人间尤物.mp4".to_string(),
            size: 78852915,
        },
        FileInfo {
            path: "威尼斯人_真人棋牌开户送777元_V8862.VIP.mp4".to_string(),
            filename: "威尼斯人_真人棋牌开户送777元_V8862.VIP.mp4".to_string(),
            size: 25480396,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 17: {:?}", result);
}

#[test]
fn sample_018() {
    // [ www.Torrenting.com ] - Red.Canyon.UNCUT.2008.BDRiP.XviD-Fi
    let info = ContentInfo {
        name: "[ www.Torrenting.com ] - Red.Canyon.UNCUT.2008.BDRiP.XviD-FiEND".to_string(),
        files: vec![
        FileInfo {
            path: "fiend-rcanyonx.avi".to_string(),
            filename: "fiend-rcanyonx.avi".to_string(),
            size: 733478912,
        },
        FileInfo {
            path: "Sample/fiend-rcanyonx-sample.avi".to_string(),
            filename: "fiend-rcanyonx-sample.avi".to_string(),
            size: 8703180,
        },
        FileInfo {
            path: "fiend-rcanyonx.nfo".to_string(),
            filename: "fiend-rcanyonx.nfo".to_string(),
            size: 10137,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 18: {:?}", result);
}

#[test]
fn sample_019() {
    // 七公主的司机.mp4
    let info = ContentInfo {
        name: "七公主的司机.mp4".to_string(),
        files: vec![
        FileInfo {
            path: "七公主的司机.mp4".to_string(),
            filename: "七公主的司机.mp4".to_string(),
            size: 1288490188,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 19: {:?}", result);
}

#[test]
fn sample_020() {
    // Alena Kravchenko Disk1
    let info = ContentInfo {
        name: "Alena Kravchenko Disk1".to_string(),
        files: vec![
        FileInfo {
            path: "DISK_#1/Уроки/Видеоуроки/UROK 6.avi".to_string(),
            filename: "UROK 6.avi".to_string(),
            size: 179411353,
        },
        FileInfo {
            path: "DISK_#1/Уроки/Видеоуроки/UROK 1.avi".to_string(),
            filename: "UROK 1.avi".to_string(),
            size: 146066636,
        },
        FileInfo {
            path: "DISK_#1/Уроки/Видеоуроки/UROK 5.avi".to_string(),
            filename: "UROK 5.avi".to_string(),
            size: 141662617,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 20: {:?}", result);
}

#[test]
fn sample_021() {
    // Neil Young Smothered
    let info = ContentInfo {
        name: "Neil Young Smothered".to_string(),
        files: vec![
        FileInfo {
            path: "... Beach/For the Turnstiles - Josh Rouse (White  Session, france- 02.03.05).mp3".to_string(),
            filename: "For the Turnstiles - Josh Rouse (White  Session, france- 02.03.05).mp3".to_string(),
            size: 136524595,
        },
        FileInfo {
            path: "1994 - Sleeps With Angels/06 - Change Your Mind.mp3".to_string(),
            filename: "06 - Change Your Mind.mp3".to_string(),
            size: 35232153,
        },
        FileInfo {
            path: "...Beach/On the Beach - Outformation (House of BLues, Orlando, FL 2-28-2009).mp3".to_string(),
            filename: "On the Beach - Outformation (House of BLues, Orlando, FL 2-28-2009).mp3".to_string(),
            size: 32505856,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Audio, "Sample 21: {:?}", result);
}

#[test]
fn sample_022() {
    // WinRAR 5.50 32Bit And 64Bit Full-Version
    let info = ContentInfo {
        name: "WinRAR 5.50 32Bit And 64Bit Full-Version".to_string(),
        files: vec![
        FileInfo {
            path: "WinRAR 5.50 32Bit And 64Bit Full-Version.exe".to_string(),
            filename: "WinRAR 5.50 32Bit And 64Bit Full-Version.exe".to_string(),
            size: 29674700,
        },
        FileInfo {
            path: "WinRAR 5.50 32Bit And 64Bit Full-Version.nfo".to_string(),
            filename: "WinRAR 5.50 32Bit And 64Bit Full-Version.nfo".to_string(),
            size: 576,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Software, "Sample 22: {:?}", result);
}

#[test]
fn sample_023() {
    // [R.G. Mechanics] This War of Mine
    let info = ContentInfo {
        name: "[R.G. Mechanics] This War of Mine".to_string(),
        files: vec![
        FileInfo {
            path: "data1.bin".to_string(),
            filename: "data1.bin".to_string(),
            size: 430650163,
        },
        FileInfo {
            path: "data2.bin".to_string(),
            filename: "data2.bin".to_string(),
            size: 235929600,
        },
        FileInfo {
            path: "Redist/vcredist_x86_2010.exe".to_string(),
            filename: "vcredist_x86_2010.exe".to_string(),
            size: 3040870,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Software, "Sample 23: {:?}", result);
}

#[test]
fn sample_024() {
    // wankzvr-charitable-donations-180_180x180_3dh_LR.mp4
    let info = ContentInfo {
        name: "wankzvr-charitable-donations-180_180x180_3dh_LR.mp4".to_string(),
        files: vec![
        FileInfo {
            path: "wankzvr-charitable-donations-180_180x180_3dh_LR.mp4".to_string(),
            filename: "wankzvr-charitable-donations-180_180x180_3dh_LR.mp4".to_string(),
            size: 6335076761,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 24: {:?}", result);
}

#[test]
fn sample_025() {
    // Marsha May - The Anal Princess 720.mp4
    let info = ContentInfo {
        name: "Marsha May - The Anal Princess 720.mp4".to_string(),
        files: vec![
        FileInfo {
            path: "Marsha May - The Anal Princess 720.mp4".to_string(),
            filename: "Marsha May - The Anal Princess 720.mp4".to_string(),
            size: 1395864371,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 25: {:?}", result);
}

#[test]
fn sample_026() {
    // Stargirl.S01E04.1080p.rus.LostFilm.TV.mkv
    let info = ContentInfo {
        name: "Stargirl.S01E04.1080p.rus.LostFilm.TV.mkv".to_string(),
        files: vec![
        FileInfo {
            path: "Stargirl.S01E04.1080p.rus.LostFilm.TV.mkv".to_string(),
            filename: "Stargirl.S01E04.1080p.rus.LostFilm.TV.mkv".to_string(),
            size: 2469606195,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 26: {:?}", result);
}

#[test]
fn sample_027() {
    // Shans.na.lyubov.2017.WEB-DL.(1080p).Getty
    let info = ContentInfo {
        name: "Shans.na.lyubov.2017.WEB-DL.(1080p).Getty".to_string(),
        files: vec![
        FileInfo {
            path: "Shans.na.lyubov.e01.2017.WEB-DL.(1080p).Getty.mkv".to_string(),
            filename: "Shans.na.lyubov.e01.2017.WEB-DL.(1080p).Getty.mkv".to_string(),
            size: 1932735283,
        },
        FileInfo {
            path: "Shans.na.lyubov.e04.2017.WEB-DL.(1080p).Getty.mkv".to_string(),
            filename: "Shans.na.lyubov.e04.2017.WEB-DL.(1080p).Getty.mkv".to_string(),
            size: 1717986918,
        },
        FileInfo {
            path: "Shans.na.lyubov.e03.2017.WEB-DL.(1080p).Getty.mkv".to_string(),
            filename: "Shans.na.lyubov.e03.2017.WEB-DL.(1080p).Getty.mkv".to_string(),
            size: 1717986918,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 27: {:?}", result);
}

#[test]
fn sample_028() {
    // Neverwinter_Nights_2_Complete.rar
    let info = ContentInfo {
        name: "Neverwinter_Nights_2_Complete.rar".to_string(),
        files: vec![
        FileInfo {
            path: "Neverwinter_Nights_2_Complete.rar".to_string(),
            filename: "Neverwinter_Nights_2_Complete.rar".to_string(),
            size: 8375186227,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Other, "Sample 28: {:?}", result);
}

#[test]
fn sample_029() {
    // The Fog 2005 Unrated 1080p Blu-Ray HEVC x265 10Bit DDP5.1 Su
    let info = ContentInfo {
        name: "The Fog 2005 Unrated 1080p Blu-Ray HEVC x265 10Bit DDP5.1 Subs KINGDOM_RG".to_string(),
        files: vec![
        FileInfo {
            path: "The Fog 2005 Unrated 1080p Blu-Ray HEVC x265 10Bit DDP5.1 Subs KINGDOM_new.mkv".to_string(),
            filename: "The Fog 2005 Unrated 1080p Blu-Ray HEVC x265 10Bit DDP5.1 Subs KINGDOM_new.mkv".to_string(),
            size: 5153960755,
        },
        FileInfo {
            path: "KINGDOM  RG.mkv".to_string(),
            filename: "KINGDOM  RG.mkv".to_string(),
            size: 2936012,
        },
        FileInfo {
            path: "Subs/ID2_[en] {SDH}.srt".to_string(),
            filename: "ID2_[en] {SDH}.srt".to_string(),
            size: 82944,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 29: {:?}", result);
}

#[test]
fn sample_030() {
    // HND-965
    let info = ContentInfo {
        name: "HND-965".to_string(),
        files: vec![
        FileInfo {
            path: "big2048.com@HND-965.mp4".to_string(),
            filename: "big2048.com@HND-965.mp4".to_string(),
            size: 5368709120,
        },
        FileInfo {
            path: "澳门银河赌场-注册免费送36元 可提款-.mp4".to_string(),
            filename: "澳门银河赌场-注册免费送36元 可提款-.mp4".to_string(),
            size: 10066329,
        },
        FileInfo {
            path: "澳门威尼斯人注册免费送48元.mp4".to_string(),
            filename: "澳门威尼斯人注册免费送48元.mp4".to_string(),
            size: 8493465,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 30: {:?}", result);
}

#[test]
fn sample_031() {
    // L'Échappée S05E24 FINAL VFQ 720p WEBRIP x264-MTLQC.mp4
    let info = ContentInfo {
        name: "L'Échappée S05E24 FINAL VFQ 720p WEBRIP x264-MTLQC.mp4".to_string(),
        files: vec![
        FileInfo {
            path: "L'Échappée S05E24 FINAL VFQ 720p WEBRIP x264-MTLQC.mp4".to_string(),
            filename: "L'Échappée S05E24 FINAL VFQ 720p WEBRIP x264-MTLQC.mp4".to_string(),
            size: 699714764,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 31: {:?}", result);
}

#[test]
fn sample_032() {
    // SpankMonster.19.10.17.Chanel.Grey.XXX.1080p.MP4-KTR[rarbg]
    let info = ContentInfo {
        name: "SpankMonster.19.10.17.Chanel.Grey.XXX.1080p.MP4-KTR[rarbg]".to_string(),
        files: vec![
        FileInfo {
            path: "spankmonster.19.10.17.chanel.grey.mp4".to_string(),
            filename: "spankmonster.19.10.17.chanel.grey.mp4".to_string(),
            size: 1610612736,
        },
        FileInfo {
            path: "spankmonster.19.10.17.chanel.grey.nfo".to_string(),
            filename: "spankmonster.19.10.17.chanel.grey.nfo".to_string(),
            size: 5222,
        },
        FileInfo {
            path: "RARBG_DO_NOT_MIRROR.exe".to_string(),
            filename: "RARBG_DO_NOT_MIRROR.exe".to_string(),
            size: 99,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 32: {:?}", result);
}

#[test]
fn sample_033() {
    // Preacher (S03) LOST
    let info = ContentInfo {
        name: "Preacher (S03) LOST".to_string(),
        files: vec![
        FileInfo {
            path: "Preacher.S03E02.WEB-DLRip.Rus.Eng.LostFilm.avi".to_string(),
            filename: "Preacher.S03E02.WEB-DLRip.Rus.Eng.LostFilm.avi".to_string(),
            size: 525231718,
        },
        FileInfo {
            path: "Preacher.S03E01.WEB-DLRip.Rus.Eng.LostFilm.avi".to_string(),
            filename: "Preacher.S03E01.WEB-DLRip.Rus.Eng.LostFilm.avi".to_string(),
            size: 524917145,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 33: {:?}", result);
}

#[test]
fn sample_034() {
    // moc.15.02.17.hollie.shields(1).mp4
    let info = ContentInfo {
        name: "moc.15.02.17.hollie.shields(1).mp4".to_string(),
        files: vec![
        FileInfo {
            path: "moc.15.02.17.hollie.shields(1).mp4".to_string(),
            filename: "moc.15.02.17.hollie.shields(1).mp4".to_string(),
            size: 579757670,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 34: {:?}", result);
}

#[test]
fn sample_035() {
    // 露逼露奶进村勾引老头给老大爷吃性药增加战力丧偶的老头很牛逼狂吃BB亮点是打完炮后用杯子接她尿喝说苦骚味源码录制
    let info = ContentInfo {
        name: "露逼露奶进村勾引老头给老大爷吃性药增加战力丧偶的老头很牛逼狂吃BB亮点是打完炮后用杯子接她尿喝说苦骚味源码录制".to_string(),
        files: vec![
        FileInfo {
            path: "露逼露奶进村勾引老头给老大爷吃性药增加战力丧偶的老头很牛逼狂吃BB亮点是打完炮后用杯子接她尿喝说苦骚味源码录制.mp4".to_string(),
            filename: "露逼露奶进村勾引老头给老大爷吃性药增加战力丧偶的老头很牛逼狂吃BB亮点是打完炮后用杯子接她尿喝说苦骚味源码录制.mp4".to_string(),
            size: 313943654,
        },
        FileInfo {
            path: "(  1024社区最新地址_2.0 ).urll".to_string(),
            filename: "(  1024社区最新地址_2.0 ).urll".to_string(),
            size: 114,
        },
        FileInfo {
            path: "1024核工厂 1024核工厂 - Powered by Discuz!.url".to_string(),
            filename: "1024核工厂 1024核工厂 - Powered by Discuz!.url".to_string(),
            size: 114,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 35: {:?}", result);
}

#[test]
fn sample_036() {
    // Where.the.Road.Runs.Out.2014.1080p.BluRay.H264.AAC-RARBG
    let info = ContentInfo {
        name: "Where.the.Road.Runs.Out.2014.1080p.BluRay.H264.AAC-RARBG".to_string(),
        files: vec![
        FileInfo {
            path: "Where.the.Road.Runs.Out.2014.1080p.BluRay.H264.AAC-RARBG.mp4".to_string(),
            filename: "Where.the.Road.Runs.Out.2014.1080p.BluRay.H264.AAC-RARBG.mp4".to_string(),
            size: 1932735283,
        },
        FileInfo {
            path: "RARBG_DO_NOT_MIRROR.exe".to_string(),
            filename: "RARBG_DO_NOT_MIRROR.exe".to_string(),
            size: 99,
        },
        FileInfo {
            path: "RARBG.txt".to_string(),
            filename: "RARBG.txt".to_string(),
            size: 30,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 36: {:?}", result);
}

#[test]
fn sample_037() {
    // [2024.10.23] 櫻坂46 10thシングル「I want tomorrow to come」[スペシャルエディ
    let info = ContentInfo {
        name: "[2024.10.23] 櫻坂46 10thシングル「I want tomorrow to come」[スペシャルエディション][FLAC]".to_string(),
        files: vec![
        FileInfo {
            path: "07. TOKYO SNOW.flac".to_string(),
            filename: "07. TOKYO SNOW.flac".to_string(),
            size: 35232153,
        },
        FileInfo {
            path: "06. 19歳のガレット.flac".to_string(),
            filename: "06. 19歳のガレット.flac".to_string(),
            size: 35127296,
        },
        FileInfo {
            path: "05. 嵐の前、世界の終わり.flac".to_string(),
            filename: "05. 嵐の前、世界の終わり.flac".to_string(),
            size: 35127296,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Audio, "Sample 37: {:?}", result);
}

#[test]
fn sample_038() {
    // 高颜值钢管舞美女第四弹 性感透视装黑丝带着面具跳舞 很是诱惑不要错过
    let info = ContentInfo {
        name: "高颜值钢管舞美女第四弹 性感透视装黑丝带着面具跳舞 很是诱惑不要错过".to_string(),
        files: vec![
        FileInfo {
            path: "高颜值钢管舞美女第四弹 性感透视装黑丝带着面具跳舞 很是诱惑不要错过.mp4".to_string(),
            filename: "高颜值钢管舞美女第四弹 性感透视装黑丝带着面具跳舞 很是诱惑不要错过.mp4".to_string(),
            size: 28835840,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 38: {:?}", result);
}

#[test]
fn sample_039() {
    // [��Ѹ��������-www.xinxl.com]-Singers-2007-11-23-YDY.rmvb
    let info = ContentInfo {
        name: "[��Ѹ��������-www.xinxl.com]-Singers-2007-11-23-YDY.rmvb".to_string(),
        files: vec![
        FileInfo {
            path: "[��Ѹ��������-www.xinxl.com]-Singers-2007-11-23-YDY.rmvb".to_string(),
            filename: "[��Ѹ��������-www.xinxl.com]-Singers-2007-11-23-YDY.rmvb".to_string(),
            size: 625265868,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 39: {:?}", result);
}

#[test]
fn sample_040() {
    // 1026-5.zip
    let info = ContentInfo {
        name: "1026-5.zip".to_string(),
        files: vec![
        FileInfo {
            path: "1026-5.zip".to_string(),
            filename: "1026-5.zip".to_string(),
            size: 258998272,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Other, "Sample 40: {:?}", result);
}

#[test]
fn sample_041() {
    // Unnamed Memory
    let info = ContentInfo {
        name: "Unnamed Memory".to_string(),
        files: vec![
        FileInfo {
            path: "[SubsPlease] Unnamed Memory - 05 (720p) [FD06448A].mkv".to_string(),
            filename: "[SubsPlease] Unnamed Memory - 05 (720p) [FD06448A].mkv".to_string(),
            size: 738931507,
        },
        FileInfo {
            path: "[SubsPlease] Unnamed Memory - 01 (720p) [3F89344A].mkv".to_string(),
            filename: "[SubsPlease] Unnamed Memory - 01 (720p) [3F89344A].mkv".to_string(),
            size: 737463500,
        },
        FileInfo {
            path: "[SubsPlease] Unnamed Memory - 03 (720p) [9090FF02].mkv".to_string(),
            filename: "[SubsPlease] Unnamed Memory - 03 (720p) [9090FF02].mkv".to_string(),
            size: 737148928,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 41: {:?}", result);
}

#[test]
fn sample_042() {
    // Aquaman And The Lost Kingdom 2023 1080p V3 Clean Cam New Aud
    let info = ContentInfo {
        name: "Aquaman And The Lost Kingdom 2023 1080p V3 Clean Cam New Audio X264 Will1869".to_string(),
        files: vec![
        FileInfo {
            path: "Aquaman And The Lost Kingdom 2023 1080p V3 Clean Cam New Audio X264 Will1869.mp4".to_string(),
            filename: "Aquaman And The Lost Kingdom 2023 1080p V3 Clean Cam New Audio X264 Will1869.mp4".to_string(),
            size: 2899102924,
        },
        FileInfo {
            path: "Sample 2.mp4".to_string(),
            filename: "Sample 2.mp4".to_string(),
            size: 44354764,
        },
        FileInfo {
            path: "Sample 1.mp4".to_string(),
            filename: "Sample 1.mp4".to_string(),
            size: 19503513,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 42: {:?}", result);
}

#[test]
fn sample_043() {
    // Maria Sadowska ‎– Początek Nocy (2020) [FLAC].rar
    let info = ContentInfo {
        name: "Maria Sadowska ‎– Początek Nocy (2020) [FLAC].rar".to_string(),
        files: vec![
        FileInfo {
            path: "Maria Sadowska ‎– Początek Nocy (2020) [FLAC].rar".to_string(),
            filename: "Maria Sadowska ‎– Początek Nocy (2020) [FLAC].rar".to_string(),
            size: 277767782,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Audio, "Sample 43: {:?}", result);
}

#[test]
fn sample_044() {
    // Porno Kid horny preteen litl3 11yo venezuelan pthc baba bebe
    let info = ContentInfo {
        name: "Porno Kid horny preteen litl3 11yo venezuelan pthc baba bebek.mpg".to_string(),
        files: vec![
        FileInfo {
            path: "Porno Kid horny preteen litl3 11yo venezuelan pthc baba bebek.mpg".to_string(),
            filename: "Porno Kid horny preteen litl3 11yo venezuelan pthc baba bebek.mpg".to_string(),
            size: 293810995,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 44: {:?}", result);
}

#[test]
fn sample_045() {
    // [lunch] Koinaka (Ch. 1-2, 4, 6-8) [English]
    let info = ContentInfo {
        name: "[lunch] Koinaka (Ch. 1-2, 4, 6-8) [English]".to_string(),
        files: vec![
        FileInfo {
            path: "143.png".to_string(),
            filename: "143.png".to_string(),
            size: 772198,
        },
        FileInfo {
            path: "141.png".to_string(),
            filename: "141.png".to_string(),
            size: 766464,
        },
        FileInfo {
            path: "063.png".to_string(),
            filename: "063.png".to_string(),
            size: 748544,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Other, "Sample 45: {:?}", result);
}

#[test]
fn sample_046() {
    // BattleBots.2015.S03E11.720p.WEB.x264-TBS[rarbg]
    let info = ContentInfo {
        name: "BattleBots.2015.S03E11.720p.WEB.x264-TBS[rarbg]".to_string(),
        files: vec![
        FileInfo {
            path: "battlebots.2015.s03e11.720p.web.x264-tbs.mkv".to_string(),
            filename: "battlebots.2015.s03e11.720p.web.x264-tbs.mkv".to_string(),
            size: 979265126,
        },
        FileInfo {
            path: "battlebots.2015.s03e11.720p.web.x264-tbs.nfo".to_string(),
            filename: "battlebots.2015.s03e11.720p.web.x264-tbs.nfo".to_string(),
            size: 52,
        },
        FileInfo {
            path: "RARBG.txt".to_string(),
            filename: "RARBG.txt".to_string(),
            size: 30,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 46: {:?}", result);
}

#[test]
fn sample_047() {
    // Bill.Burr.Walk.Your.Way.Out.2017.720p.NF.WEBRip.H264.AAC-PRi
    let info = ContentInfo {
        name: "Bill.Burr.Walk.Your.Way.Out.2017.720p.NF.WEBRip.H264.AAC-PRiNCE[PRiME]".to_string(),
        files: vec![
        FileInfo {
            path: "Bill.Burr.Walk.Your.Way.Out.2017.720p.NF.WEBRip.H264.AAC-PRiNCE[PRiME].mkv".to_string(),
            filename: "Bill.Burr.Walk.Your.Way.Out.2017.720p.NF.WEBRip.H264.AAC-PRiNCE[PRiME].mkv".to_string(),
            size: 1825361100,
        },
        FileInfo {
            path: "Torrent Downloaded From ExtraTorrent.cc.txt".to_string(),
            filename: "Torrent Downloaded From ExtraTorrent.cc.txt".to_string(),
            size: 446,
        },
        FileInfo {
            path: "Please Support!! Read More!!!.txt".to_string(),
            filename: "Please Support!! Read More!!!.txt".to_string(),
            size: 232,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 47: {:?}", result);
}

#[test]
fn sample_048() {
    // A.Study.in.Terror.1965.720p.BluRay.H264.AAC-RARBG
    let info = ContentInfo {
        name: "A.Study.in.Terror.1965.720p.BluRay.H264.AAC-RARBG".to_string(),
        files: vec![
        FileInfo {
            path: "A.Study.in.Terror.1965.720p.BluRay.H264.AAC-RARBG.mp4".to_string(),
            filename: "A.Study.in.Terror.1965.720p.BluRay.H264.AAC-RARBG.mp4".to_string(),
            size: 1181116006,
        },
        FileInfo {
            path: "RARBG.COM.mp4".to_string(),
            filename: "RARBG.COM.mp4".to_string(),
            size: 1016729,
        },
        FileInfo {
            path: "A.Study.in.Terror.1965.720p.BluRay.H264.AAC-RARBG.nfo".to_string(),
            filename: "A.Study.in.Terror.1965.720p.BluRay.H264.AAC-RARBG.nfo".to_string(),
            size: 3891,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 48: {:?}", result);
}

#[test]
fn sample_049() {
    // А.Фет - Вечера и ночи (Listik9).mp3
    let info = ContentInfo {
        name: "А.Фет - Вечера и ночи (Listik9).mp3".to_string(),
        files: vec![
        FileInfo {
            path: "А.Фет - Вечера и ночи (Listik9).mp3".to_string(),
            filename: "А.Фет - Вечера и ночи (Listik9).mp3".to_string(),
            size: 44459622,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Audio, "Sample 49: {:?}", result);
}

#[test]
fn sample_050() {
    // Marc dorcel - Pornochic Collection 0 - 30
    let info = ContentInfo {
        name: "Marc dorcel - Pornochic Collection 0 - 30".to_string(),
        files: vec![
        FileInfo {
            path: "Movie/Pornochic 24 - Ariel & Lola.mp4".to_string(),
            filename: "Pornochic 24 - Ariel & Lola.mp4".to_string(),
            size: 3865470566,
        },
        FileInfo {
            path: "Movie/Pornochic 25 - Anissa Kate.mp4".to_string(),
            filename: "Pornochic 25 - Anissa Kate.mp4".to_string(),
            size: 3435973836,
        },
        FileInfo {
            path: "Movie/Pornochic 21 - Aleska & Angelika.mp4".to_string(),
            filename: "Pornochic 21 - Aleska & Angelika.mp4".to_string(),
            size: 3328599654,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 50: {:?}", result);
}

#[test]
fn sample_051() {
    // Cathedral In Flames (2019 - 2021)
    let info = ContentInfo {
        name: "Cathedral In Flames (2019 - 2021)".to_string(),
        files: vec![
        FileInfo {
            path: "...ang Me High & Bury Me Deep/04 - Hang me high and bury me deep (Desperado).mp3".to_string(),
            filename: "04 - Hang me high and bury me deep (Desperado).mp3".to_string(),
            size: 19608371,
        },
        FileInfo {
            path: "Singles/2020 - Desperado/01 Desperado.mp3".to_string(),
            filename: "01 Desperado.mp3".to_string(),
            size: 14050918,
        },
        FileInfo {
            path: "2021 - Hang Me High & Bury Me Deep/07 - Dia de los muertos.mp3".to_string(),
            filename: "07 - Dia de los muertos.mp3".to_string(),
            size: 13212057,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Audio, "Sample 51: {:?}", result);
}

#[test]
fn sample_052() {
    // [ThZu.Cc]fc2ppv_1161006
    let info = ContentInfo {
        name: "[ThZu.Cc]fc2ppv_1161006".to_string(),
        files: vec![
        FileInfo {
            path: "[ThZu.Cc]fc2ppv_1161006.mp4".to_string(),
            filename: "[ThZu.Cc]fc2ppv_1161006.mp4".to_string(),
            size: 2362232012,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 52: {:?}", result);
}

#[test]
fn sample_053() {
    // 今日的网漫第09集.mp4
    let info = ContentInfo {
        name: "今日的网漫第09集.mp4".to_string(),
        files: vec![
        FileInfo {
            path: "今日的网漫第09集.mp4".to_string(),
            filename: "今日的网漫第09集.mp4".to_string(),
            size: 616772403,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 53: {:?}", result);
}

#[test]
fn sample_054() {
    // UMSO-103
    let info = ContentInfo {
        name: "UMSO-103".to_string(),
        files: vec![
        FileInfo {
            path: "UMSO-103.mp4".to_string(),
            filename: "UMSO-103.mp4".to_string(),
            size: 940048384,
        },
        FileInfo {
            path: "xue0117@草榴社区@最新地址.lnk".to_string(),
            filename: "xue0117@草榴社区@最新地址.lnk".to_string(),
            size: 1013,
        },
        FileInfo {
            path: "xue0117@草榴社区@最新地址.URL".to_string(),
            filename: "xue0117@草榴社区@最新地址.URL".to_string(),
            size: 74,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 54: {:?}", result);
}

#[test]
fn sample_055() {
    // Angel.Wicky.artporn.mp4
    let info = ContentInfo {
        name: "Angel.Wicky.artporn.mp4".to_string(),
        files: vec![
        FileInfo {
            path: "Angel.Wicky.artporn.mp4".to_string(),
            filename: "Angel.Wicky.artporn.mp4".to_string(),
            size: 337431756,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 55: {:?}", result);
}

#[test]
fn sample_056() {
    // [Fanbox] 阿戈魔AGM [Decensored].zip
    let info = ContentInfo {
        name: "[Fanbox] 阿戈魔AGM [Decensored].zip".to_string(),
        files: vec![
        FileInfo {
            path: "[Fanbox] 阿戈魔AGM [Decensored].zip".to_string(),
            filename: "[Fanbox] 阿戈魔AGM [Decensored].zip".to_string(),
            size: 8267812044,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Other, "Sample 56: {:?}", result);
}

#[test]
fn sample_057() {
    // DASS-014_6K-C
    let info = ContentInfo {
        name: "DASS-014_6K-C".to_string(),
        files: vec![
        FileInfo {
            path: "@bo99.tv_DASS-014_C.mp4".to_string(),
            filename: "@bo99.tv_DASS-014_C.mp4".to_string(),
            size: 5583457484,
        },
        FileInfo {
            path: "张信哲的成功致富方法.mp4".to_string(),
            filename: "张信哲的成功致富方法.mp4".to_string(),
            size: 35127296,
        },
        FileInfo {
            path: "华语No.1 AV大平台.mp4".to_string(),
            filename: "华语No.1 AV大平台.mp4".to_string(),
            size: 32820428,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 57: {:?}", result);
}

#[test]
fn sample_058() {
    // datpiff-mixtape-ma4810e6
    let info = ContentInfo {
        name: "datpiff-mixtape-ma4810e6".to_string(),
        files: vec![
        FileInfo {
            path: "ma4810e6-mixtape.zip".to_string(),
            filename: "ma4810e6-mixtape.zip".to_string(),
            size: 56518246,
        },
        FileInfo {
            path: "02 - Want It All.mp3".to_string(),
            filename: "02 - Want It All.mp3".to_string(),
            size: 11848908,
        },
        FileInfo {
            path: "03 - Rexton Rawlston Fernando Gordon.mp3".to_string(),
            filename: "03 - Rexton Rawlston Fernando Gordon.mp3".to_string(),
            size: 10695475,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Audio, "Sample 58: {:?}", result);
}

#[test]
fn sample_059() {
    // Jimmy.Kimmel.2022.05.23.Tom.Hiddleston.720p.WEB.H264-JEBAITE
    let info = ContentInfo {
        name: "Jimmy.Kimmel.2022.05.23.Tom.Hiddleston.720p.WEB.H264-JEBAITED[rarbg]".to_string(),
        files: vec![
        FileInfo {
            path: "jimmy.kimmel.2022.05.23.tom.hiddleston.720p.web.h264-jebaited.mkv".to_string(),
            filename: "jimmy.kimmel.2022.05.23.tom.hiddleston.720p.web.h264-jebaited.mkv".to_string(),
            size: 1932735283,
        },
        FileInfo {
            path: "RARBG_DO_NOT_MIRROR.exe".to_string(),
            filename: "RARBG_DO_NOT_MIRROR.exe".to_string(),
            size: 99,
        },
        FileInfo {
            path: "jimmy.kimmel.2022.05.23.tom.hiddleston.720p.web.h264-jebaited.nfo".to_string(),
            filename: "jimmy.kimmel.2022.05.23.tom.hiddleston.720p.web.h264-jebaited.nfo".to_string(),
            size: 62,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 59: {:?}", result);
}

#[test]
fn sample_060() {
    // ATFB300AVI
    let info = ContentInfo {
        name: "ATFB300AVI".to_string(),
        files: vec![
        FileInfo {
            path: "ATFB300.avi".to_string(),
            filename: "ATFB300.avi".to_string(),
            size: 1181116006,
        },
        FileInfo {
            path: "論壇文宣/waikeung最佳成人交友園.jpg".to_string(),
            filename: "waikeung最佳成人交友園.jpg".to_string(),
            size: 294502,
        },
        FileInfo {
            path: "ATFB300B.jpg".to_string(),
            filename: "ATFB300B.jpg".to_string(),
            size: 171417,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 60: {:?}", result);
}

#[test]
fn sample_061() {
    // Опять совещание  Как превратить пустые обсуждения в эффектив
    let info = ContentInfo {
        name: "Опять совещание  Как превратить пустые обсуждения в эффективные".to_string(),
        files: vec![
        FileInfo {
            path: "opyat-soveschanie.rtf".to_string(),
            filename: "opyat-soveschanie.rtf".to_string(),
            size: 36280729,
        },
        FileInfo {
            path: "opyat-soveschanie.fb2".to_string(),
            filename: "opyat-soveschanie.fb2".to_string(),
            size: 3565158,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Book, "Sample 61: {:?}", result);
}

#[test]
fn sample_062() {
    // baby in diapers cbaby child cjhildporn PedoMom 11YR yumiko15
    let info = ContentInfo {
        name: "baby in diapers cbaby child cjhildporn PedoMom 11YR yumiko15yo preteen pedofilo.mpg".to_string(),
        files: vec![
        FileInfo {
            path: "...n diapers cbaby child cjhildporn PedoMom 11YR yumiko15yo preteen pedofilo.mpg".to_string(),
            filename: "baby in diapers cbaby child cjhildporn PedoMom 11YR yumiko15yo preteen pedofilo.mpg".to_string(),
            size: 268330598,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 62: {:?}", result);
}

#[test]
fn sample_063() {
    // MomPOV.Julia.hardcore.pov.milf.anal.mp4
    let info = ContentInfo {
        name: "MomPOV.Julia.hardcore.pov.milf.anal.mp4".to_string(),
        files: vec![
        FileInfo {
            path: "MomPOV.Julia.hardcore.pov.milf.anal.mp4".to_string(),
            filename: "MomPOV.Julia.hardcore.pov.milf.anal.mp4".to_string(),
            size: 1029387059,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 63: {:?}", result);
}

#[test]
fn sample_064() {
    // Botched.S08E14.Lip.Filler.Killer.720p.AMZN.WEB-DL.DDP2.0.H.2
    let info = ContentInfo {
        name: "Botched.S08E14.Lip.Filler.Killer.720p.AMZN.WEB-DL.DDP2.0.H.264-NTb[TGx]".to_string(),
        files: vec![
        FileInfo {
            path: "Botched.S08E14.Lip.Filler.Killer.720p.AMZN.WEB-DL.DDP2.0.H.264-NTb.mkv".to_string(),
            filename: "Botched.S08E14.Lip.Filler.Killer.720p.AMZN.WEB-DL.DDP2.0.H.264-NTb.mkv".to_string(),
            size: 1503238553,
        },
        FileInfo {
            path: "Botched.S08E14.Lip.Filler.Killer.720p.AMZN.WEB-DL.DDP2.0.H.264-NTb.nfo".to_string(),
            filename: "Botched.S08E14.Lip.Filler.Killer.720p.AMZN.WEB-DL.DDP2.0.H.264-NTb.nfo".to_string(),
            size: 4812,
        },
        FileInfo {
            path: "[TGx]Downloaded from torrentgalaxy.to .txt".to_string(),
            filename: "[TGx]Downloaded from torrentgalaxy.to .txt".to_string(),
            size: 479,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 64: {:?}", result);
}

#[test]
fn sample_065() {
    // 偷拍楼下南京某大学妹子洗澡,附带抖音截图,感慨美颜技术实在太强大了
    let info = ContentInfo {
        name: "偷拍楼下南京某大学妹子洗澡,附带抖音截图,感慨美颜技术实在太强大了".to_string(),
        files: vec![
        FileInfo {
            path: "偷拍楼下南京某大学妹子洗澡,附带抖音截图,感慨美颜技术实在太强大了.mp4".to_string(),
            filename: "偷拍楼下南京某大学妹子洗澡,附带抖音截图,感慨美颜技术实在太强大了.mp4".to_string(),
            size: 144284057,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 65: {:?}", result);
}

#[test]
fn sample_066() {
    // (成年コミック) [完顔阿骨打] (2001-07-30) 神奈月の姉妹 マッド薬剤師砂恵.zip
    let info = ContentInfo {
        name: "(成年コミック) [完顔阿骨打] (2001-07-30) 神奈月の姉妹 マッド薬剤師砂恵.zip".to_string(),
        files: vec![
        FileInfo {
            path: "(成年コミック) [完顔阿骨打] (2001-07-30) 神奈月の姉妹 マッド薬剤師砂恵.zip".to_string(),
            filename: "(成年コミック) [完顔阿骨打] (2001-07-30) 神奈月の姉妹 マッド薬剤師砂恵.zip".to_string(),
            size: 182871654,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Book, "Sample 66: {:?}", result);
}

#[test]
fn sample_067() {
    // Черепашки.2014.720p.LEONARDO_[scarabey.org].mkv
    let info = ContentInfo {
        name: "Черепашки.2014.720p.LEONARDO_[scarabey.org].mkv".to_string(),
        files: vec![
        FileInfo {
            path: "Черепашки.2014.720p.LEONARDO_[scarabey.org].mkv".to_string(),
            filename: "Черепашки.2014.720p.LEONARDO_[scarabey.org].mkv".to_string(),
            size: 3758096384,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 67: {:?}", result);
}

#[test]
fn sample_068() {
    // Lesbea.19.12.29.Vanessa.Decker.And.Lexi.Dona.XXX.1080p.HEVC.
    let info = ContentInfo {
        name: "Lesbea.19.12.29.Vanessa.Decker.And.Lexi.Dona.XXX.1080p.HEVC.x265.PRT".to_string(),
        files: vec![
        FileInfo {
            path: "Lesbea.19.12.29.Vanessa.Decker.And.Lexi.Dona.XXX.1080p.HEVC.x265.PRT.mp4".to_string(),
            filename: "Lesbea.19.12.29.Vanessa.Decker.And.Lexi.Dona.XXX.1080p.HEVC.x265.PRT.mp4".to_string(),
            size: 269169459,
        },
        FileInfo {
            path: "Provided by PornRips.to.nfo".to_string(),
            filename: "Provided by PornRips.to.nfo".to_string(),
            size: 47,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 68: {:?}", result);
}

#[test]
fn sample_069() {
    // Dawicontrol-FORCED-881x64-15.000_5.00_old-drp.zip
    let info = ContentInfo {
        name: "Dawicontrol-FORCED-881x64-15.000_5.00_old-drp.zip".to_string(),
        files: vec![
        FileInfo {
            path: "Dawicontrol-FORCED-881x64-15.000_5.00_old-drp.zip".to_string(),
            filename: "Dawicontrol-FORCED-881x64-15.000_5.00_old-drp.zip".to_string(),
            size: 32358,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Software, "Sample 69: {:?}", result);
}

#[test]
fn sample_070() {
    // Cum On My Hairy Pussy 09 [porno][www.lokotorrents.com]
    let info = ContentInfo {
        name: "Cum On My Hairy Pussy 09 [porno][www.lokotorrents.com]".to_string(),
        files: vec![
        FileInfo {
            path: "Cum On My Hairy Pussy 09 cd1.avi".to_string(),
            filename: "Cum On My Hairy Pussy 09 cd1.avi".to_string(),
            size: 733583769,
        },
        FileInfo {
            path: "Cum On My Hairy Pussy 09 cd2.avi".to_string(),
            filename: "Cum On My Hairy Pussy 09 cd2.avi".to_string(),
            size: 726348595,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 70: {:?}", result);
}

#[test]
fn sample_071() {
    // 38.zip
    let info = ContentInfo {
        name: "38.zip".to_string(),
        files: vec![
        FileInfo {
            path: "38.zip".to_string(),
            filename: "38.zip".to_string(),
            size: 15414067,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Other, "Sample 71: {:?}", result);
}

#[test]
fn sample_072() {
    // Every Day 2018 Movies 720p HDRip x264 5.1 with Sample ☻rDX☻
    let info = ContentInfo {
        name: "Every Day 2018 Movies 720p HDRip x264 5.1 with Sample ☻rDX☻".to_string(),
        files: vec![
        FileInfo {
            path: "Every Day 2018 Movies 720p HDRip x264 5.1 ☻rDX☻.mkv".to_string(),
            filename: "Every Day 2018 Movies 720p HDRip x264 5.1 ☻rDX☻.mkv".to_string(),
            size: 938894950,
        },
        FileInfo {
            path: "Sample ~ Every Day 2018 Movies 720p HDRip x264 5.1 ☻rDX☻.mkv".to_string(),
            filename: "Sample ~ Every Day 2018 Movies 720p HDRip x264 5.1 ☻rDX☻.mkv".to_string(),
            size: 9751756,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 72: {:?}", result);
}

#[test]
fn sample_073() {
    // sprd1198c
    let info = ContentInfo {
        name: "sprd1198c".to_string(),
        files: vec![
        FileInfo {
            path: "sprd1198c.mp4".to_string(),
            filename: "sprd1198c.mp4".to_string(),
            size: 728236032,
        },
        FileInfo {
            path: "uuf83.mp4".to_string(),
            filename: "uuf83.mp4".to_string(),
            size: 55155097,
        },
        FileInfo {
            path: "啪啪啪會所.mp4".to_string(),
            filename: "啪啪啪會所.mp4".to_string(),
            size: 2306867,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 73: {:?}", result);
}

#[test]
fn sample_074() {
    // Idiotsitter - Temporada 1 [HDTV 720p][Cap.101][AC3 5.1 Españ
    let info = ContentInfo {
        name: "Idiotsitter - Temporada 1 [HDTV 720p][Cap.101][AC3 5.1 Español Castellano]".to_string(),
        files: vec![
        FileInfo {
            path: "Idiotsitter720p_101_WWW.NEWPCT1.COM.mkv".to_string(),
            filename: "Idiotsitter720p_101_WWW.NEWPCT1.COM.mkv".to_string(),
            size: 933337497,
        },
        FileInfo {
            path: "www.DIVXATOPE.com.url".to_string(),
            filename: "www.DIVXATOPE.com.url".to_string(),
            size: 877568,
        },
        FileInfo {
            path: "www.DIVXATOPE1.com.url".to_string(),
            filename: "www.DIVXATOPE1.com.url".to_string(),
            size: 877568,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 74: {:?}", result);
}

#[test]
fn sample_075() {
    // Peppermint.2018.720p.BluRay.x264-SPARKS
    let info = ContentInfo {
        name: "Peppermint.2018.720p.BluRay.x264-SPARKS".to_string(),
        files: vec![
        FileInfo {
            path: "peppermint.2018.720p.bluray.x264-sparks.r00".to_string(),
            filename: "peppermint.2018.720p.bluray.x264-sparks.r00".to_string(),
            size: 50017075,
        },
        FileInfo {
            path: "peppermint.2018.720p.bluray.x264-sparks.r01".to_string(),
            filename: "peppermint.2018.720p.bluray.x264-sparks.r01".to_string(),
            size: 50017075,
        },
        FileInfo {
            path: "peppermint.2018.720p.bluray.x264-sparks.r02".to_string(),
            filename: "peppermint.2018.720p.bluray.x264-sparks.r02".to_string(),
            size: 50017075,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 75: {:?}", result);
}

#[test]
fn sample_076() {
    // [4K城堡]飞向月球.全5集.2019.2160p.HEVC.10bit.HDR.DD5.1.国语中字[www.4kcb
    let info = ContentInfo {
        name: "[4K城堡]飞向月球.全5集.2019.2160p.HEVC.10bit.HDR.DD5.1.国语中字[www.4kcb.com]".to_string(),
        files: vec![
        FileInfo {
            path: "[4K城堡]飞向月球.第1集.2019 2160p.HEVC.10bit.HDR.DD5.1.国语中字[www.4kcb.com].ts".to_string(),
            filename: "[4K城堡]飞向月球.第1集.2019 2160p.HEVC.10bit.HDR.DD5.1.国语中字[www.4kcb.com].ts".to_string(),
            size: 8160437862,
        },
        FileInfo {
            path: "[4K城堡]飞向月球.第5集.2019 2160p.HEVC.10bit.HDR.DD5.1.国语中字[www.4kcb.com].ts".to_string(),
            filename: "[4K城堡]飞向月球.第5集.2019 2160p.HEVC.10bit.HDR.DD5.1.国语中字[www.4kcb.com].ts".to_string(),
            size: 8160437862,
        },
        FileInfo {
            path: "[4K城堡]飞向月球.第4集.2019 2160p.HEVC.10bit.HDR.DD5.1.国语中字[www.4kcb.com].ts".to_string(),
            filename: "[4K城堡]飞向月球.第4集.2019 2160p.HEVC.10bit.HDR.DD5.1.国语中字[www.4kcb.com].ts".to_string(),
            size: 8160437862,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 76: {:?}", result);
}

#[test]
fn sample_077() {
    // Beliy.plen.2006.XviD.BDRip.ExKinoRay.avi
    let info = ContentInfo {
        name: "Beliy.plen.2006.XviD.BDRip.ExKinoRay.avi".to_string(),
        files: vec![
        FileInfo {
            path: "Beliy.plen.2006.XviD.BDRip.ExKinoRay.avi".to_string(),
            filename: "Beliy.plen.2006.XviD.BDRip.ExKinoRay.avi".to_string(),
            size: 2362232012,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 77: {:?}", result);
}

#[test]
fn sample_078() {
    // Thomas_Cathcart_-_There_Is_No_God_and_Mary_Is_His_Mother
    let info = ContentInfo {
        name: "Thomas_Cathcart_-_There_Is_No_God_and_Mary_Is_His_Mother".to_string(),
        files: vec![
        FileInfo {
            path: "There Is No God and Mary Is His Mother-Part02.mp3".to_string(),
            filename: "There Is No God and Mary Is His Mother-Part02.mp3".to_string(),
            size: 31457280,
        },
        FileInfo {
            path: "There Is No God and Mary Is His Mother-Part04.mp3".to_string(),
            filename: "There Is No God and Mary Is His Mother-Part04.mp3".to_string(),
            size: 30094131,
        },
        FileInfo {
            path: "There Is No God and Mary Is His Mother-Part01.mp3".to_string(),
            filename: "There Is No God and Mary Is His Mother-Part01.mp3".to_string(),
            size: 26424115,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Audio, "Sample 78: {:?}", result);
}

#[test]
fn sample_079() {
    // Jan Jett - Student Bodies 02.mp4
    let info = ContentInfo {
        name: "Jan Jett - Student Bodies 02.mp4".to_string(),
        files: vec![
        FileInfo {
            path: "Jan Jett - Student Bodies 02.mp4".to_string(),
            filename: "Jan Jett - Student Bodies 02.mp4".to_string(),
            size: 229638144,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 79: {:?}", result);
}

#[test]
fn sample_080() {
    // Soul Asylum - No Fun Intended (2013)
    let info = ContentInfo {
        name: "Soul Asylum - No Fun Intended (2013)".to_string(),
        files: vec![
        FileInfo {
            path: "02 - Love Will Tear Us Apart.mp3".to_string(),
            filename: "02 - Love Will Tear Us Apart.mp3".to_string(),
            size: 6396313,
        },
        FileInfo {
            path: "03 - Shakin' Street.mp3".to_string(),
            filename: "03 - Shakin' Street.mp3".to_string(),
            size: 5033164,
        },
        FileInfo {
            path: "01 - Attacking the Beat.mp3".to_string(),
            filename: "01 - Attacking the Beat.mp3".to_string(),
            size: 3040870,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Audio, "Sample 80: {:?}", result);
}

#[test]
fn sample_081() {
    // GirlCum - Sybil - So Wet
    let info = ContentInfo {
        name: "GirlCum - Sybil - So Wet".to_string(),
        files: vec![
        FileInfo {
            path: "jav20s8.com GirlCum - Sybil - So Wet.mp4".to_string(),
            filename: "jav20s8.com GirlCum - Sybil - So Wet.mp4".to_string(),
            size: 918552576,
        },
        FileInfo {
            path: "N房间的精彩直播 只有你想不到的刺激uur78.mp4".to_string(),
            filename: "N房间的精彩直播 只有你想不到的刺激uur78.mp4".to_string(),
            size: 22020096,
        },
        FileInfo {
            path: "美女直播.mp4".to_string(),
            filename: "美女直播.mp4".to_string(),
            size: 10590617,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 81: {:?}", result);
}

#[test]
fn sample_082() {
    // 与公公偷腥的儿媳 시아버지를 농락한 며느리.2023.HD1080P.韩语中字
    let info = ContentInfo {
        name: "与公公偷腥的儿媳 시아버지를 농락한 며느리.2023.HD1080P.韩语中字".to_string(),
        files: vec![
        FileInfo {
            path: "与公公偷腥的儿媳 시아버지를 농락한 며느리.2023.HD1080P.韩语中字.mp4".to_string(),
            filename: "与公公偷腥的儿媳 시아버지를 농락한 며느리.2023.HD1080P.韩语中字.mp4".to_string(),
            size: 1932735283,
        },
        FileInfo {
            path: "(_ 最新bt合集发布}.mht".to_string(),
            filename: "(_ 最新bt合集发布}.mht".to_string(),
            size: 1004748,
        },
        FileInfo {
            path: "_____padding_file_1_如果您看到此文件，请升级到BitComet(比特彗星)0.85或以上版本____".to_string(),
            filename: "_____padding_file_1_如果您看到此文件，请升级到BitComet(比特彗星)0.85或以上版本____".to_string(),
            size: 522137,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 82: {:?}", result);
}

#[test]
fn sample_083() {
    // Southern.Charm.S06E00.How.They.Got.Here.720p.WEB.x264-TBS[ra
    let info = ContentInfo {
        name: "Southern.Charm.S06E00.How.They.Got.Here.720p.WEB.x264-TBS[rarbg]".to_string(),
        files: vec![
        FileInfo {
            path: "southern.charm.s06e00.how.they.got.here.720p.web.x264-tbs.mkv".to_string(),
            filename: "southern.charm.s06e00.how.they.got.here.720p.web.x264-tbs.mkv".to_string(),
            size: 425826713,
        },
        FileInfo {
            path: "RARBG_DO_NOT_MIRROR.exe".to_string(),
            filename: "RARBG_DO_NOT_MIRROR.exe".to_string(),
            size: 99,
        },
        FileInfo {
            path: "southern.charm.s06e00.how.they.got.here.720p.web.x264-tbs.nfo".to_string(),
            filename: "southern.charm.s06e00.how.they.got.here.720p.web.x264-tbs.nfo".to_string(),
            size: 69,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 83: {:?}", result);
}

#[test]
fn sample_084() {
    // POtHS - The Word - 044 - Mark Biltz - The Feasts of the Lord
    let info = ContentInfo {
        name: "POtHS - The Word - 044 - Mark Biltz - The Feasts of the Lord".to_string(),
        files: vec![
        FileInfo {
            path: "SIGNS pt45/POGMark.avi".to_string(),
            filename: "POGMark.avi".to_string(),
            size: 914987417,
        },
        FileInfo {
            path: "Shorts/Who Is Israel.avi".to_string(),
            filename: "Who Is Israel.avi".to_string(),
            size: 896847052,
        },
        FileInfo {
            path: "SIGNS pt45/The Moon Shaking.avi".to_string(),
            filename: "The Moon Shaking.avi".to_string(),
            size: 693842739,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 84: {:?}", result);
}

#[test]
fn sample_085() {
    // Přepadení ze vzduchu S01
    let info = ContentInfo {
        name: "Přepadení ze vzduchu S01".to_string(),
        files: vec![
        FileInfo {
            path: "S01E008_Příliš blízko ke slunci.mkv".to_string(),
            filename: "S01E008_Příliš blízko ke slunci.mkv".to_string(),
            size: 2362232012,
        },
        FileInfo {
            path: "S01E005_Vyvolat tsunami.mkv".to_string(),
            filename: "S01E005_Vyvolat tsunami.mkv".to_string(),
            size: 2147483648,
        },
        FileInfo {
            path: "S01E006_Akce.mkv".to_string(),
            filename: "S01E006_Akce.mkv".to_string(),
            size: 2147483648,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 85: {:?}", result);
}

#[test]
fn sample_086() {
    // Miles Davis & Quincy Jones - Miles & Quincy Live at Montreux
    let info = ContentInfo {
        name: "Miles Davis & Quincy Jones - Miles & Quincy Live at Montreux (Live Version) (1993 Jazz) [Flac 16-44]".to_string(),
        files: vec![
        FileInfo {
            path: "16. Miles Davis & Quincy Jones - Solea (Live Version) (Live Album Version).flac".to_string(),
            filename: "16. Miles Davis & Quincy Jones - Solea (Live Version) (Live Album Version).flac".to_string(),
            size: 81579212,
        },
        FileInfo {
            path: "...vis & Quincy Jones - Blues for Pablo (Live Version) (Live Album Version).flac".to_string(),
            filename: "09. Miles Davis & Quincy Jones - Blues for Pablo (Live Version) (Live Album Version).flac".to_string(),
            size: 39216742,
        },
        FileInfo {
            path: "11. Miles Davis & Quincy Jones - Orgone (Live Version) (Live Album Version).flac".to_string(),
            filename: "11. Miles Davis & Quincy Jones - Orgone (Live Version) (Live Album Version).flac".to_string(),
            size: 30828134,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Audio, "Sample 86: {:?}", result);
}

#[test]
fn sample_087() {
    // Ade Chichi Tsurime de Midara de Yabai Hisho
    let info = ContentInfo {
        name: "Ade Chichi Tsurime de Midara de Yabai Hisho".to_string(),
        files: vec![
        FileInfo {
            path: "adechichi.iso".to_string(),
            filename: "adechichi.iso".to_string(),
            size: 592864870,
        },
        FileInfo {
            path: "01.png".to_string(),
            filename: "01.png".to_string(),
            size: 3250585,
        },
        FileInfo {
            path: "レーベル.png".to_string(),
            filename: "レーベル.png".to_string(),
            size: 902860,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Other, "Sample 87: {:?}", result);
}

#[test]
fn sample_088() {
    // x5h5.com 良家夫妻在家里直播激情啪啪，露脸了大哥吃奶子舌吻，口交大鸡巴舔逼道具抽插啥都会，各种体位爆草真刺激
    let info = ContentInfo {
        name: "x5h5.com 良家夫妻在家里直播激情啪啪，露脸了大哥吃奶子舌吻，口交大鸡巴舔逼道具抽插啥都会，各种体位爆草真刺激".to_string(),
        files: vec![
        FileInfo {
            path: "x5h5.com 良家夫妻在家里直播激情啪啪，露脸了大哥吃奶子舌吻，口交大鸡巴舔逼道具抽插啥都会，各种体位爆草真刺激.mp4".to_string(),
            filename: "x5h5.com 良家夫妻在家里直播激情啪啪，露脸了大哥吃奶子舌吻，口交大鸡巴舔逼道具抽插啥都会，各种体位爆草真刺激.mp4".to_string(),
            size: 1069232947,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 88: {:?}", result);
}

#[test]
fn sample_089() {
    // Helt.Perfekt.S02E01-AndyNor
    let info = ContentInfo {
        name: "Helt.Perfekt.S02E01-AndyNor".to_string(),
        files: vec![
        FileInfo {
            path: "Helt.Perfekt.S02E01-AndyNor.mp4".to_string(),
            filename: "Helt.Perfekt.S02E01-AndyNor.mp4".to_string(),
            size: 285946675,
        },
        FileInfo {
            path: "ReadMe.txt".to_string(),
            filename: "ReadMe.txt".to_string(),
            size: 831,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 89: {:?}", result);
}

#[test]
fn sample_090() {
    // (C94) [GOLD KOMAN SEX (Baksheesh AT)] Kasshoku Kokumaro Funn
    let info = ContentInfo {
        name: "(C94) [GOLD KOMAN SEX (Baksheesh AT)] Kasshoku Kokumaro Funnyuu Maid Stardust Genius Kanketsuhen [En".to_string(),
        files: vec![
        FileInfo {
            path: "...ardust Genius Kanketsuhen [English] [Hive-san] (Updated 2019-03-26)-1280x.zip".to_string(),
            filename: "(C94) [GOLD KOMAN SEX (Baksheesh AT)] Kasshoku Kokumaro Funnyuu Maid Stardust Genius Kanketsuhen [English] [Hive-san] (Updated 2019-03-26)-1280x.zip".to_string(),
            size: 20761804,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Book, "Sample 90: {:?}", result);
}

#[test]
fn sample_091() {
    // The.Witcher.3.Wild.Hunt.Free.DLC.Program.v2.0.0.45-GOG
    let info = ContentInfo {
        name: "The.Witcher.3.Wild.Hunt.Free.DLC.Program.v2.0.0.45-GOG".to_string(),
        files: vec![
        FileInfo {
            path: "setup_the_witcher3_dlc1-16_2.0.0.45.exe".to_string(),
            filename: "setup_the_witcher3_dlc1-16_2.0.0.45.exe".to_string(),
            size: 547880960,
        },
        FileInfo {
            path: "README.txt".to_string(),
            filename: "README.txt".to_string(),
            size: 603,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Software, "Sample 91: {:?}", result);
}

#[test]
fn sample_092() {
    // The Drafter (The Peri Reed Chronicles #1) by Kim Harrison (e
    let info = ContentInfo {
        name: "The Drafter (The Peri Reed Chronicles #1) by Kim Harrison (epub) {OLQ} [BЯ]".to_string(),
        files: vec![
        FileInfo {
            path: "The Drafter (The Peri Reed Chronicles #1) by Kim Harrison.epub.epub".to_string(),
            filename: "The Drafter (The Peri Reed Chronicles #1) by Kim Harrison.epub.epub".to_string(),
            size: 462745,
        },
        FileInfo {
            path: "cover.jpg".to_string(),
            filename: "cover.jpg".to_string(),
            size: 22528,
        },
        FileInfo {
            path: "BookRangers.nfo".to_string(),
            filename: "BookRangers.nfo".to_string(),
            size: 5324,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Book, "Sample 92: {:?}", result);
}

#[test]
fn sample_093() {
    // 야후(윤태호 원작).iso
    let info = ContentInfo {
        name: "야후(윤태호 원작).iso".to_string(),
        files: vec![
        FileInfo {
            path: "야후(윤태호 원작).iso".to_string(),
            filename: "야후(윤태호 원작).iso".to_string(),
            size: 252077670,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Other, "Sample 93: {:?}", result);
}

#[test]
fn sample_094() {
    // turner_video_1510
    let info = ContentInfo {
        name: "turner_video_1510".to_string(),
        files: vec![
        FileInfo {
            path: "1510.ia.mp4".to_string(),
            filename: "1510.ia.mp4".to_string(),
            size: 10905190,
        },
        FileInfo {
            path: "1510.mp4".to_string(),
            filename: "1510.mp4".to_string(),
            size: 10905190,
        },
        FileInfo {
            path: "history/files/1510.mp4.~1~".to_string(),
            filename: "1510.mp4.~1~".to_string(),
            size: 10905190,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 94: {:?}", result);
}

#[test]
fn sample_095() {
    // IENF-160
    let info = ContentInfo {
        name: "IENF-160".to_string(),
        files: vec![
        FileInfo {
            path: "bbs2048.org@IENF-160.mp4".to_string(),
            filename: "bbs2048.org@IENF-160.mp4".to_string(),
            size: 5583457484,
        },
        FileInfo {
            path: "妹妹在精彩表演 ———-哥哥快来大饱眼福uuf39.com.mp4".to_string(),
            filename: "妹妹在精彩表演 ———-哥哥快来大饱眼福uuf39.com.mp4".to_string(),
            size: 21495808,
        },
        FileInfo {
            path: "妹妹直播,可以指揮表演 A57X.mp4".to_string(),
            filename: "妹妹直播,可以指揮表演 A57X.mp4".to_string(),
            size: 20971520,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 95: {:?}", result);
}

#[test]
fn sample_096() {
    // 24484_Derby_p2_avc_hd.mp4
    let info = ContentInfo {
        name: "24484_Derby_p2_avc_hd.mp4".to_string(),
        files: vec![
        FileInfo {
            path: "24484_Derby_p2_avc_hd.mp4".to_string(),
            filename: "24484_Derby_p2_avc_hd.mp4".to_string(),
            size: 2147483648,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 96: {:?}", result);
}

#[test]
fn sample_097() {
    // HEYZO-2592-FHD
    let info = ContentInfo {
        name: "HEYZO-2592-FHD".to_string(),
        files: vec![
        FileInfo {
            path: "bbs2048.org@heyzo_hd_2592_full.mp4".to_string(),
            filename: "bbs2048.org@heyzo_hd_2592_full.mp4".to_string(),
            size: 2254857830,
        },
        FileInfo {
            path: "妹妹在精彩表演 ———-哥哥快来大饱眼福uuf39.com.mp4".to_string(),
            filename: "妹妹在精彩表演 ———-哥哥快来大饱眼福uuf39.com.mp4".to_string(),
            size: 21495808,
        },
        FileInfo {
            path: "妹妹直播,可以指揮表演 A57X.mp4".to_string(),
            filename: "妹妹直播,可以指揮表演 A57X.mp4".to_string(),
            size: 20971520,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 97: {:?}", result);
}

#[test]
fn sample_098() {
    // Дзюдо.Мужчины до 81 кг.mkv
    let info = ContentInfo {
        name: "Дзюдо.Мужчины до 81 кг.mkv".to_string(),
        files: vec![
        FileInfo {
            path: "Дзюдо.Мужчины до 81 кг.mkv".to_string(),
            filename: "Дзюдо.Мужчины до 81 кг.mkv".to_string(),
            size: 3865470566,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 98: {:?}", result);
}

#[test]
fn sample_099() {
    // [HD Uncensored] FC2 PPV 1132267 【個人撮影】顔出し/うみ 19歳/セーラー服が似合う清純
    let info = ContentInfo {
        name: "[HD Uncensored] FC2 PPV 1132267 【個人撮影】顔出し/うみ 19歳/セーラー服が似合う清純派/フェラさせまくり生ハメしまくりの約60分/大量中出しでフィニッシュｗｗｗ [".to_string(),
        files: vec![
        FileInfo {
            path: "FC2-PPV-1132267.mp4".to_string(),
            filename: "FC2-PPV-1132267.mp4".to_string(),
            size: 1825361100,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 99: {:?}", result);
}

#[test]
fn sample_100() {
    // [SubsPlease] Boruto - Naruto Next Generations - 180 (720p) [
    let info = ContentInfo {
        name: "[SubsPlease] Boruto - Naruto Next Generations - 180 (720p) [3993C1C9].mkv".to_string(),
        files: vec![
        FileInfo {
            path: "[SubsPlease] Boruto - Naruto Next Generations - 180 (720p) [3993C1C9].mkv".to_string(),
            filename: "[SubsPlease] Boruto - Naruto Next Generations - 180 (720p) [3993C1C9].mkv".to_string(),
            size: 745852108,
        }
        ],
    };
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, Medium::Video, "Sample 100: {:?}", result);
}

