#ifndef DataFormats_ForwardDetId_HGCScintillatorDetId_H
#define DataFormats_ForwardDetId_HGCScintillatorDetId_H 1

#include <iosfwd>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "FWCore/Utilities/interface/Exception.h"

/* \brief description of the bit assigment
   [0:8]   iphi index wrt x-axis on +z side
   [9:16]  |radius| index (starting from a minimum radius depending on type)
   [17:21] Layer #
   [22]    Trigger(1)/Detector(0) cell
   [23:24] Reserved for future extension
   [25:25] z-side (0 for +z; 1 for -z)
   [26:27] Type (0 fine divisions of scintillators;
                 1 coarse divisions of scintillators)
   [28:31] Detector type (HGCalHSc)
*/

class HGCScintillatorDetId : public DetId {
public:
  /** Create a null cellid*/
  constexpr HGCScintillatorDetId() : DetId() {}
  
  /** Create cellid from raw id (0=invalid tower id) */
  constexpr HGCScintillatorDetId(uint32_t rawid) : DetId(rawid) {}
  
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  constexpr HGCScintillatorDetId(int type, int layer, int radius, int phi, bool trigger = false)
    : DetId(HGCalHSc, ForwardEmpty) {
    int zside = (radius < 0) ? 1 : 0;
    int itrig = trigger ? 1 : 0;
    int radiusAbs = std::abs(radius);
    id_ |= (((type & kHGCalTypeMask) << kHGCalTypeOffset) | ((zside & kHGCalZsideMask) << kHGCalZsideOffset) |
            ((itrig & kHGCalTriggerMask) << kHGCalTriggerOffset) | ((layer & kHGCalLayerMask) << kHGCalLayerOffset) |
            ((radiusAbs & kHGCalRadiusMask) << kHGCalRadiusOffset) | ((phi & kHGCalPhiMask) << kHGCalPhiOffset));
  }

  /** Constructor from a generic cell id */
  constexpr HGCScintillatorDetId(const DetId& gen){
    if (!gen.null()) {
      if (gen.det() != HGCalHSc) {
        throw cms::Exception("Invalid DetId")
            << "Cannot initialize HGCScintillatorDetId from " << std::hex << gen.rawId() << std::dec;
      }
    }
    id_ = gen.rawId();
  }

  /** Assignment from a generic cell id */
  constexpr HGCScintillatorDetId& operator=(const DetId& gen){
    if (!gen.null()) {
      if (gen.det() != HGCalHSc) {
        throw cms::Exception("Invalid DetId")
            << "Cannot assign HGCScintillatorDetId from " << std::hex << gen.rawId() << std::dec;
      }
    }
    id_ = gen.rawId();
    return (*this);
  }

  /** Converter for a geometry cell id */
  constexpr HGCScintillatorDetId geometryCell() const {
    if (trigger()) {
      return HGCScintillatorDetId(type(), layer(), iradiusTrigger(), iphiTrigger(), false);
    } else {
      return HGCScintillatorDetId(type(), layer(), iradius(), iphi(), false);
    }
  }

  /// get the subdetector
  constexpr DetId::Detector subdet() const { return det(); }

  /// get the type
  constexpr int type() const { return (id_ >> kHGCalTypeOffset) & kHGCalTypeMask; }

  /// get the z-side of the cell (1/-1)
  constexpr int zside() const { return (((id_ >> kHGCalZsideOffset) & kHGCalZsideMask) ? -1 : 1); }

  /// get the layer #
  constexpr int layer() const { return (id_ >> kHGCalLayerOffset) & kHGCalLayerMask; }

  /// get the eta index
  constexpr int iradiusAbs() const {
    if (trigger())
      return (2 * ((id_ >> kHGCalRadiusOffset) & kHGCalRadiusMask));
    else
      return ((id_ >> kHGCalRadiusOffset) & kHGCalRadiusMask);
  }
  constexpr int iradius() const { return zside() * iradiusAbs(); }
  constexpr int ietaAbs() const { return iradiusAbs(); }
  constexpr int ieta() const { return zside() * ietaAbs(); }

  /// get the phi index
  constexpr int iphi() const {
    if (trigger())
      return (2 * ((id_ >> kHGCalPhiOffset) & kHGCalPhiMask));
    else
      return ((id_ >> kHGCalPhiOffset) & kHGCalPhiMask);
  }

  constexpr std::pair<int, int> ietaphi() const { return std::pair<int, int>(ieta(), iphi()); }
  constexpr std::pair<int, int> iradiusphi() const { return std::pair<int, int>(iradius(), iphi()); }

  /// trigger or detector cell
  std::vector<HGCScintillatorDetId> detectorCells() const;
  constexpr bool trigger() const { return (((id_ >> kHGCalTriggerOffset) & kHGCalTriggerMask) == 1); }
  constexpr HGCScintillatorDetId triggerCell() const {
    if (trigger())
      return HGCScintillatorDetId(type(), layer(), iradius(), iphi(), true);
    else
      return HGCScintillatorDetId(type(), layer(), iradiusTrigger(), iphiTrigger(), true);
  }

  /// consistency check : no bits left => no overhead
  constexpr bool isEE() const { return false; }
  constexpr bool isHE() const { return true; }
  constexpr bool isForward() const { return true; }

  static const HGCScintillatorDetId Undefined;

public:
  static const int kHGCalPhiOffset = 0;
  static const int kHGCalPhiMask = 0x1FF;
  static const int kHGCalRadiusOffset = 9;
  static const int kHGCalRadiusMask = 0xFF;
  static const int kHGCalLayerOffset = 17;
  static const int kHGCalLayerMask = 0x1F;
  static const int kHGCalTriggerOffset = 22;
  static const int kHGCalTriggerMask = 0x1;
  static const int kHGCalZsideOffset = 25;
  static const int kHGCalZsideMask = 0x1;
  static const int kHGCalTypeOffset = 26;
  static const int kHGCalTypeMask = 0x3;

  constexpr int iradiusTriggerAbs() const {
    if (trigger())
      return ((iradiusAbs() + 1) / 2);
    else
      return iradiusAbs();
  }
  
  constexpr int iradiusTrigger() const { return zside() * iradiusTriggerAbs(); }
  
  constexpr int iphiTrigger() const {
    if (trigger())
      return ((iphi() + 1) / 2);
    else
      return iphi();
  }
};

std::ostream& operator<<(std::ostream& s, const HGCScintillatorDetId& id);

#endif
