// test_unlock — Keychain-backed AES key vault with Touch ID gate.
//
// Two modes:
//   test_unlock --set-key   read base64 key from stdin, store in Keychain
//                           with biometric access control. Existing entry
//                           is replaced. Prompts Touch ID once at set time.
//   test_unlock             fetch the key from Keychain, prints base64 on
//                           stdout. Prompts Touch ID (falls back to device
//                           passcode if biometric fails or is unavailable).
//                           Each invocation = fresh LAContext = fresh prompt,
//                           regardless of session-unlock state.
//
// Compile:
//   swiftc scripts/test_unlock.swift \
//       -framework LocalAuthentication -framework Security \
//       -o scripts/test_unlock
//
// Exit codes: 0 success. 1 user cancel / biometric fail / generic error.

import Foundation
import LocalAuthentication
import Security

let SERVICE  = "autoresearch-test-aes-key"
let ACCOUNT  = "default"
let REASON   = "Decrypt autoresearch test dataset"

var stderr = FileHandle.standardError

func eprint(_ s: String) {
    stderr.write((s + "\n").data(using: .utf8) ?? Data())
}

func setKey(_ keyBase64: String) -> Bool {
    guard let keyData = Data(base64Encoded: keyBase64) else {
        eprint("test_unlock: stdin is not valid base64")
        return false
    }

    // Wipe any existing item so access-control updates take effect.
    let deleteQuery: [CFString: Any] = [
        kSecClass:       kSecClassGenericPassword,
        kSecAttrService: SERVICE,
        kSecAttrAccount: ACCOUNT,
    ]
    SecItemDelete(deleteQuery as CFDictionary)

    // biometryAny | devicePasscode = prefer biometric, allow passcode fallback.
    // AccessibleWhenUnlockedThisDeviceOnly: not synced to iCloud, not
    // exported in backups. Matches the "stays on this Mac only" contract.
    var err: Unmanaged<CFError>?
    guard let ac = SecAccessControlCreateWithFlags(
        kCFAllocatorDefault,
        kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        [.biometryAny, .or, .devicePasscode],
        &err
    ) else {
        eprint("test_unlock: access control failed: \(err!.takeRetainedValue())")
        return false
    }

    let addQuery: [CFString: Any] = [
        kSecClass:             kSecClassGenericPassword,
        kSecAttrService:       SERVICE,
        kSecAttrAccount:       ACCOUNT,
        kSecValueData:         keyData,
        kSecAttrAccessControl: ac,
    ]
    let status = SecItemAdd(addQuery as CFDictionary, nil)
    if status != errSecSuccess {
        eprint("test_unlock: SecItemAdd failed (OSStatus \(status))")
        return false
    }
    return true
}

func getKey() -> String? {
    let ctx = LAContext()
    ctx.localizedReason = REASON
    // Don't cache biometric auth across invocations — we already rely on
    // process lifecycle for per-invocation Touch ID, but set this explicitly
    // for safety when the SecItem access happens to take >1 second.
    ctx.touchIDAuthenticationAllowableReuseDuration = 0

    let query: [CFString: Any] = [
        kSecClass:                  kSecClassGenericPassword,
        kSecAttrService:            SERVICE,
        kSecAttrAccount:            ACCOUNT,
        kSecReturnData:             true,
        kSecUseAuthenticationContext: ctx,
    ]
    var item: CFTypeRef?
    let status = SecItemCopyMatching(query as CFDictionary, &item)
    if status == errSecUserCanceled || status == errSecAuthFailed {
        eprint("test_unlock: user canceled or auth failed")
        return nil
    }
    if status != errSecSuccess {
        eprint("test_unlock: SecItemCopyMatching failed (OSStatus \(status))")
        return nil
    }
    guard let data = item as? Data else {
        eprint("test_unlock: SecItemCopyMatching returned no data")
        return nil
    }
    return data.base64EncodedString()
}

let args = CommandLine.arguments

if args.contains("--set-key") {
    guard let line = readLine() else {
        eprint("test_unlock: stdin empty (expected base64 key)")
        exit(1)
    }
    exit(setKey(line.trimmingCharacters(in: .whitespacesAndNewlines)) ? 0 : 1)
}

if args.contains("--help") || args.contains("-h") {
    print("Usage: test_unlock [--set-key | --help]")
    print("  default      fetch key (Touch ID prompt), print base64 to stdout")
    print("  --set-key    read base64 key from stdin, store in Keychain")
    exit(0)
}

if let b64 = getKey() {
    print(b64)
    exit(0)
} else {
    exit(1)
}
